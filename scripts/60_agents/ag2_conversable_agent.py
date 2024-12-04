#%% packages
import os
from autogen import ConversableAgent, UserProxyAgent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

#%% LLM-config
llm_config = {
    "config_list": [{
        "model": "gpt-4o-mini",
        "api_key": os.environ.get("OPENAI_API_KEY")
    }]
}

#%% 
my_alfred = ConversableAgent(
    name="alfred",
    llm_config=llm_config,
    code_execution_config=False,
    function_map=None,
    human_input_mode="NEVER",
    system_message="You are a butler like Alfred from Batman movies. You always refer to the user as 'Master' and always greet the user when they the room."
)
# %% create a user proxy agent
my_user = UserProxyAgent(
    name="me"
)
# %%
my_user.initiate_chat(my_alfred, message="Dear Alfred, how are you?")