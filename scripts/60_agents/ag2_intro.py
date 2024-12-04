#%% packages
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import os

#%% LLM configuration
llm_config = config_list_from_json(env_or_file="OAI_CONFIG_LIST")


# %% set up the agents
assistant = AssistantAgent(
    name = "assistant",
    llm_config={"config_list": llm_config}
)

#%%
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding",
                           "use_docker": False})

#%%
user_proxy.initiate_chat(recipient=assistant,
                         message="Plot a chart of ETH and SOL price change YTD")
# %%
