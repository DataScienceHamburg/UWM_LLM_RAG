#%% packages
from swarm import Swarm, Agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from pprint import pprint

#%% 
client = Swarm()
# %% our first agent
agent = Agent(name = "my_first_agent",
              instructions="You are a helpful assistant that can answer questions and help with tasks.")

#%% interaction with the agent
messages = [
    {"role": "user",
     "content": "Hello, what is OpenAI Swarm?"}
]
response = client.run(agent=agent,
           messages=messages)



# %%
response.model_dump()

#%% get the model response
pprint(response.messages[-1]['content'])