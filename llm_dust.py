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


def list_agents():
    url = f"{dust_url}/api/v1/w/{wld}/assistant/agent_configurations"
    headers = {
        "Authorization": f"Bearer {dust_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        agents = response.json()
        for agent in agents["agentConfigurations"]:
            yield {"name": agent["name"], "sId": agent["sId"]}
    except requests.exceptions.RequestException as e:
        # Just return without yielding if there's an error
        return


@llm.hookimpl
def register_models(register):
    for agent_info in list_agents():
        register(
            Dust(agent_info["name"], agent_info["sId"])
        )


class Dust(llm.KeyModel):
    def __init__(self, name, sId):
        self.model_id = name
        self.agent_id = sId
        super().__init__()

    def execute(self, prompt, stream, response, conversation, key):
        return [f"hello from {self.model_id} {self.agent_id}: {prompt.prompt}"]
