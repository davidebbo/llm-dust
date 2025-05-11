import json
import os
import requests
import llm
from dotenv import load_dotenv
import time

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

def get_conversation_events(conversation_id, timeout=30):
    """
    Stream conversation events until completion or timeout.
    
    Args:
        conversation_id: The ID of the conversation
        timeout: Maximum time to wait in seconds
    
    Yields:
        Each event as it arrives
    """
    url = f"{dust_url}/api/v1/w/{wld}/assistant/conversations/{conversation_id}/events"
    
    start_time = time.time()
    
    with requests.get(url, headers=get_dust_headers(), stream=True) as response:
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                # print(decoded_line)
                if decoded_line == "data: done":
                    print("Stream finished.")
                    break
                event_data = json.loads(decoded_line[5:]) # skip the 'data:' prefix
                yield event_data["data"]
                
            # Check timeout
            if time.time() - start_time > timeout:
                print("Timeout reached, stopping event stream.")
                break
                
            # Small delay to prevent CPU spinning
            time.sleep(0.1)

def get_message_events(conversation_id, message_id):
    url = f"{dust_url}/api/v1/w/{wld}/assistant/conversations/{conversation_id}/messages/{message_id}/events"
    
    with requests.get(url, headers=get_dust_headers(), stream=True) as response:
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                # print(decoded_line)
                if decoded_line == "data: done":
                    print("Stream finished.")
                    break
                event_data = json.loads(decoded_line[5:]) # skip the 'data:' prefix
                yield event_data["data"]

def add_to_conversation(agent_id, user_prompt):
    global conversationId
    url = f"{dust_url}/api/v1/w/{wld}/assistant/conversations/{conversationId}/messages"
    data = {
        "content": user_prompt,
        "mentions": [{"configurationId": agent_id}],
        "context": {
            "username": "dust-cli-user",  # Or any other identifier
            "timezone": "Europe/Paris",  # Or dynamically get timezone
        },
        "blocking": True,
    }

    try:
        response = requests.post(url, headers=get_dust_headers(), json=data)
        response.raise_for_status()
        response_data = response.json()

        agent_reply = None
        if response_data.get("agentMessages"):
            for message in response_data["agentMessages"]:
                if message.get("type") == "agent_message" and "content" in message:
                    agent_reply = message["content"]
                    break
                if agent_reply:
                    break

        if agent_reply:
            print(f"Agent ({agent_id}): {agent_reply}")
        else:
            print("No agent reply found in the response.")
            # print("Full response for debugging:")
            # import json
            # print(json.dumps(response_data, indent=2))

    except requests.exceptions.RequestException as e:
        print(f"Error asking agent {agent_id}: {e}")
        if e.response is not None:
            print(f"Response content: {e.response.text}")
    except json.JSONDecodeError:
        print("Error decoding JSON response from server.")




@llm.hookimpl
def register_models(register):
    for agent_info in list_agents():
        register(
            Dust(agent_info["name"], agent_info["sId"])
        )


@llm.hookimpl
def register_commands(cli):
    @cli.command(name="agents")
    def agents():
        for agent_info in list_agents():
            print(f"{agent_info["name"]}: {agent_info["description"]}")
        
class Dust(llm.KeyModel):
    def __init__(self, name, sId):
        self.model_id = name
        self.agent_id = sId
        self.can_stream = True
        super().__init__()

    def execute(self, prompt, stream, response, conversation, key):
        conversation_id = create_new_conversation(self.agent_id, prompt.prompt)
        # conversation_id = "IrsNT9G1xd"

        # print(f"Conversation ID: {conversation_id}")

        for event in get_conversation_events(conversation_id):
            match event["type"]:
                case "user_message_new":
                    # content = event["message"]["content"]
                    # print(f"User: {content}")
                    pass
                case "agent_message_new":
                    messageId = event["message"]["sId"]
                    # print(f"Agent Message ID: {messageId}")
                
                    for message_event in get_message_events(conversation_id, messageId):
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
                    print(f"Conversation Title: {event['title']}")
                case "_":
                    print(f"Unknown event type: {event['type']}")
