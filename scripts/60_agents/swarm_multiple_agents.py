#%% packages
from swarm import Swarm, Agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from pprint import pprint
# %%
client = Swarm()

#%% define the function to be used by the agents
def transfer_to_polish_agent():
    """Transfer to the German Agent."""
    return polish_agent

def transfer_to_english_agent():
    """Transfer to the English Agent."""
    return english_agent

#%% define the agents
english_agent = Agent(
    name= "English Agent",
    instructions="You are a helpful agent and only speak in English.",
    functions=[transfer_to_polish_agent]
)

polish_agent = Agent(
    name="Polish Agent",
    instructions="You are a helpful agent and only speak in Polish.",
    functions=[transfer_to_english_agent]
)


# %% start the conversation
messages = [
    {"role": "user",
     "content": "Potrzebuję pomocy z rezerwacją"}
]

response = client.run(
    agent=english_agent,
    messages=messages
)

# %%
pprint(response.messages[-1]["content"])
# %%
response.model_dump()