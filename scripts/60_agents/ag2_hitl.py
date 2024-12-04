#%% packages
from autogen import ConversableAgent, UserProxyAgent
from dotenv import load_dotenv, find_dotenv
import os
#  load the environment variables
load_dotenv(find_dotenv(usecwd=True))
# %% llm config_list
config_list = {"config_list": [
    {"model": "gpt-4o-mini", 
     "temperature": 0.9, 
     "api_key": os.environ.get("OPENAI_API_KEY")}]}

# %% idea:
# one agent has a 3-digit number, e.g. 250
# the other one guesses the number, e.g. 453
# the agent with the secret knowledge returns the number of correct digits
# agent_with_secret_knowledge = ConversableAgent(
#     name="agent_with_secret_knowledge", 
#     system_message="You have a secret 3-digit number. The other agent is guessing the number. You return the number of correct digits. Even if the digit is not at the right position, count it. Example: guess 123, secret 781 - the 1 is correctly guessed.",
#     llm_config=config_list,
#     human_input_mode="NEVER",
#     # stop after 3 rounds
#     max_consecutive_auto_reply=3
# )

# agent_guessing = ConversableAgent(
#     name="agent_guessing", 
#     system_message="You are guessing the 3-digit number. Always provide a 3-digit number as 'Guess: ...'. You ask the other agent for hints.",
#     llm_config=config_list,
#     human_input_mode="NEVER",
#     # stop after 3 rounds
#     max_consecutive_auto_reply=3
# )
# # %%
# result = agent_with_secret_knowledge.initiate_chat(recipient=agent_guessing, message="I have a secret 3-digit number. Guess it. I will tell you the number of correct digits.")
# %% Human in the loop
secret_number = '007'
agent_with_secret_knowledge = ConversableAgent(
    name="agent_with_secret_knowledge", 
    system_message=f"You have a secret 3-digit number: {secret_number}. The other agent is guessing the number. You return the number of correct digits.",
    llm_config=config_list,
    human_input_mode="NEVER",
    # max_consecutive_auto_reply=1,
    is_termination_msg=lambda msg: f"{secret_number}" in msg['content']
)

agent_guessing = ConversableAgent(
    name="agent_guessing", 
    system_message="You are guessing the 3-digit number. Always provide a 3-digit number as 'Guess: ...'. You ask the other agent for hints.",
    llm_config=config_list,
    human_input_mode="ALWAYS",
    # stop after 3 rounds
    # max_consecutive_auto_reply=1,
    is_termination_msg=lambda msg: f"{secret_number}" in msg['content']
)
                                               

result = agent_with_secret_knowledge.initiate_chat(
    recipient=agent_guessing, 
    message="I have a secret 3-digit number. Start guessing. You ask for hints. I will tell you the number of correct digits.")


# %%
