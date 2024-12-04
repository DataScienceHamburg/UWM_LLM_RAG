#%% packages
from autogen import ConversableAgent
from dotenv import load_dotenv, find_dotenv
import os
# load the environment variables
load_dotenv(find_dotenv(usecwd=True))

#%% LLM-configuration
llm_config = {"config_list": [
        {"model": "gpt-4o-mini", 
         "temperature": 0.8,
         "api_key": os.environ.get("OPENAI_API_KEY")}        
    ]}
# %% set up the two different agents
jack_the_flat_earther = ConversableAgent(
    name="jack",
    system_message="You believe that the earth is flat and try to convince other of this.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

alice_normal = ConversableAgent(
    name="alice",
    system_message="You are a person who believes that the earth is round.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    # max_consecutive_auto_reply=5
)

# %%
result = jack_the_flat_earther.initiate_chat(
    recipient=alice_normal,
    message="Hello, how can you not see that the earth is flat?",
    max_turns=5
)
# %%
