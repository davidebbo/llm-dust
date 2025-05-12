import json
import os
import requests
import llm
from dotenv import load_dotenv

load_dotenv()

dust_token = os.getenv("DUST_TOKEN")
wld = os.getenv("WLD")
space_id = os.getenv("SPACE_ID")
dsId = os.getenv("DSID")
dust_url = os.getenv("DUST_URL")


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

def create_new_conversation(agent_id, user_prompt):
    url = f"{dust_url}/api/v1/w/{wld}/assistant/conversations"
    data = {
        "message": {
            "content": user_prompt,
            "mentions": [{"configurationId": agent_id}],
            "context": {
                "username": "dust-cli-user",
                "timezone": "Europe/Paris",
            },
        },
        "blocking": False, # Because we want to stream the response
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
        "blocking": False, # Because we want to stream the response
    }

    response = requests.post(url, headers=get_dust_headers(), json=data)
    response.raise_for_status()

def get_events_helper(url):
    with requests.get(url, headers=get_dust_headers(), stream=True) as response:
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                # print(decoded_line)
                if decoded_line == "data: done":
                    # print("Stream finished.")
                    break
                event_data = json.loads(decoded_line[5:]) # skip the 'data:' prefix
                yield event_data["data"]

def get_conversation_events(conversation_id, timeout=30):
    url = f"{dust_url}/api/v1/w/{wld}/assistant/conversations/{conversation_id}/events"
    return get_events_helper(url)

def get_message_events(conversation_id, message_id):
    """Retrieve and yield events from a specific message in a conversation."""
    url = f"{dust_url}/api/v1/w/{wld}/assistant/conversations/{conversation_id}/messages/{message_id}/events"
    return get_events_helper(url)


@llm.hookimpl
def register_models(register):
    """Register all available Dust agents as models."""
    for agent_info in list_agents():
        register(
            Dust(agent_info["name"], agent_info["sId"])
        )


@llm.hookimpl
def register_commands(cli):
    @cli.command(name="dust-agents")
    def agents():
        """List all available Dust agents with their descriptions."""
        for agent_info in list_agents():
            print(f"{agent_info['name']}: {agent_info['description']}")
        

class Dust(llm.KeyModel):
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
            self.conversation_id = create_new_conversation(self.agent_id, prompt.prompt)
        else:
            add_to_conversation(self.agent_id, prompt.prompt, self.conversation_id)

        for event in get_conversation_events(self.conversation_id):
            match event["type"]:
                case "user_message_new":
                    # content = event["message"]["content"]
                    # print(f"User: {content}")
                    pass
                case "agent_message_new":
                    message_id = event["message"]["sId"]
                    if message_id in self.processed_message_ids:
                        continue

                    self.processed_message_ids.add(message_id)

                    # print(f"Agent Message ID: {new_message_id}")

                
                    for message_event in get_message_events(self.conversation_id, message_id):
                        match message_event["type"]:
                            case "retrieval_params":
                                print(f"Retrieval Params: {message_event['dataSources']}")
                            case "dust_app_run_params":
                                print(f"Dust App Run Params: {message_event['action']}")
                            case "dust_app_run_block":
                                print(f"Dust App Run Block: {message_event['action']}")
                            case "agent_action_success":
                                print(f"Agent Action Success: {message_event['action']}")
                            case "generation_tokens":
                                yield message_event["text"]
                            case "agent_message_success":
                                # print("Agent Message Success!")
                                return
                            case "agent_error":
                                print(f"Agent Error: {message_event['error']}")
                            case "_":
                                print(f"Unknown message event type: {message_event['type']}")

                case "conversation_title":
                    # print(f"Conversation Title: {event['title']}")
                    pass
                case "_":
                    print(f"Unknown event type: {event['type']}")
