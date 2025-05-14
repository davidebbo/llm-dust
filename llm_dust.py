from pathlib import Path
from typing import Optional
import llm
import json
import os
import requests
from dotenv import load_dotenv


@llm.hookimpl
def register_models(register):
    """Register all available Dust agents as models."""
    for agent_info in list_agents():
        register(Dust(agent_info["name"], agent_info["sId"]))


@llm.hookimpl
def register_commands(cli):
    @cli.command(name="dust-agents")
    def agents():
        """List all available Dust agents with their descriptions."""
        for agent_info in list_agents():
            print(f"{agent_info['name']}: {agent_info['description']}")


class Dust(llm.KeyModel):
    attachment_types = {
        "image/png",
        "image/jpeg",
        "image/gif",
        "text/plain",
        "text/csv",
        "application/pdf",
    }

    def __init__(self, name, agent_id):
        self.model_id = name
        self.agent_id = agent_id
        self.can_stream = True
        self.conversation_id = None
        self.processed_message_ids = set()
        super().__init__()

    def execute(self, prompt, stream, response, conversation, key):
        """Execute a prompt against the Dust agent and stream the response."""
        message_id = None
        if not self.conversation_id:
            # Create a new conversation if one doesn't exist yet
            # It feels dirty to store the id in the class, but not sure how else to do it
            # unless we can get the command line to preserve state
            self.conversation_id = create_new_conversation(self.agent_id, prompt)
        else:
            add_to_conversation(self.agent_id, prompt.prompt, self.conversation_id)

        for event in get_conversation_events(self.conversation_id):
            match event["type"]:
                case "user_message_new":
                    # We don't care about the user message events
                    pass
                case "agent_message_new":
                    message_id = event["message"]["sId"]
                    if message_id in self.processed_message_ids:
                        continue

                    self.processed_message_ids.add(message_id)

                    for message_event in get_message_events(
                        self.conversation_id, message_id
                    ):
                        match message_event["type"]:
                            case "generation_tokens":
                                yield message_event["text"]
                            case "agent_message_success":
                                # print("Agent Message Success!")
                                return
                            case (
                                "retrieval_params"
                                | "dust_app_run_params"
                                | "dust_app_run_block"
                                | "agent_action_success"
                            ):
                                print(f"Message Event: {message_event['type']}")
                            case "agent_error":
                                print(f"Agent Error: {message_event['error']}")
                            case "_":
                                print(
                                    f"Unknown message event type: {message_event['type']}"
                                )

                case "conversation_title":
                    # print(f"Conversation Title: {event['title']}")
                    pass
                case "_":
                    print(f"Unknown event type: {event['type']}")


load_dotenv()


dust_token = os.getenv("DUST_TOKEN")
dust_url = os.getenv("DUST_URL")
wld = os.getenv("WLD")


def get_dust_headers():
    return {
        "Authorization": f"Bearer {dust_token}",
        "Content-Type": "application/json",
    }


def list_agents():
    url = f"{dust_url}/api/v1/w/{wld}/assistant/agent_configurations"

    response = requests.get(url, headers=get_dust_headers())
    response.raise_for_status()
    agents = response.json()
    for agent in agents["agentConfigurations"]:
        yield agent


def create_new_conversation(agent_id, prompt: llm.Prompt):
    url = f"{dust_url}/api/v1/w/{wld}/assistant/conversations"
    data = {
        "message": {
            "content": prompt.prompt,
            "mentions": [{"configurationId": agent_id}],
            "context": {
                "username": "dust-cli-user",
                "timezone": "Europe/Paris",
            },
        },
        "contentFragments": [
            {
                "title": os.path.basename(attachment.path),
                "fileId": upload_file_and_get_attachment_id(attachment),
            }
            for attachment in prompt.attachments
        ],
        "blocking": False,  # Because we want to stream the response
    }

    response = requests.post(url, headers=get_dust_headers(), json=data)
    response.raise_for_status()
    return response.json()["conversation"]["sId"]


def add_to_conversation(agent_id, user_prompt, conversationId):
    url = f"{dust_url}/api/v1/w/{wld}/assistant/conversations/{conversationId}/messages"
    data = {
        "content": user_prompt,
        "mentions": [{"configurationId": agent_id}],
        "context": {
            "username": "dust-cli-user",
            "timezone": "Europe/Paris",
        },
        "blocking": False,  # Because we want to stream the response
    }

    response = requests.post(url, headers=get_dust_headers(), json=data)
    response.raise_for_status()


def get_events_helper(url):
    with requests.get(url, headers=get_dust_headers(), stream=True) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                # print(decoded_line)
                if decoded_line == "data: done":
                    # print("Stream finished.")
                    break
                event_data = json.loads(decoded_line[5:])  # skip the 'data:' prefix
                yield event_data["data"]


def get_conversation_events(conversation_id, timeout=30):
    url = f"{dust_url}/api/v1/w/{wld}/assistant/conversations/{conversation_id}/events"
    return get_events_helper(url)


def get_message_events(conversation_id, message_id):
    """Retrieve and yield events from a specific message in a conversation."""
    url = f"{dust_url}/api/v1/w/{wld}/assistant/conversations/{conversation_id}/messages/{message_id}/events"
    return get_events_helper(url)


def upload_file_and_get_attachment_id(attachment: llm.Attachment) -> None:
    # Get a URL that the file can be uploaded to
    upload_url = get_file_upload_url(attachment.path, attachment.type)

    # Upload the file to that URL
    return upload_file(attachment.path, upload_url, attachment.type)


def get_file_upload_url(file_path: str, content_type: str) -> Optional[str]:
    path = Path(file_path)

    if not path.exists():
        print(f"File not found: {file_path}")
        return None

    try:
        file_size = path.stat().st_size
    except OSError as e:
        print(f"Error accessing file {file_path}: {e}")
        return None

    url = f"{dust_url}/api/v1/w/{wld}/files"
    data = {
        "contentType": content_type,
        "fileName": path.name,
        "fileSize": file_size,
        "useCase": "conversation",
    }

    try:
        response = requests.post(url, headers=get_dust_headers(), json=data)
        response.raise_for_status()
        return response.json()["file"]["uploadUrl"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting upload URL for {file_path}: {e}")
        return None


def upload_file(file_path: str, upload_url: str, content_type: str) -> Optional[str]:
    path = Path(file_path)
    try:
        with path.open("rb") as file:
            files = {"file": (path.name, file, content_type)}
            response = requests.post(
                upload_url,
                headers={"Authorization": f"Bearer {dust_token}"},
                files=files,
            )
            response.raise_for_status()
            return response.json()["file"]["sId"]
    except requests.exceptions.RequestException as e:
        print(f"Error uploading file {file_path}: {e}")
        return None
