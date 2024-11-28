#%% packages
from swarm import Swarm, Agent
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from pprint import pprint
import wikipedia

# %% wikipedia tools
def get_wikipedia_summary(query: str):
    """Get the summary of a Wikipedia article."""
    return wikipedia.page(query).summary

def search_wikipedia(query: str):
    """Search for a Wikipedia article"""
    return wikipedia.search(query)

#%% Set up the agent
wikipedia_agent = Agent(
    name="Wikipedia Agent",
    instructions="""
    You are a helpful assistant that can answer questions about Wikipedia by finding and analyzing the content of Wikipedia articles.
    You follow these steps:
    1. Find out what the user is interested in
    2. extract keywords
    3. Search for the keywords in Wikipedia using search_wikipedia
    4. From the results list, pick the most relevant article and search with get_wikipedia_summary 
    5. If you find an answer, stop and answer. If not, continue with step 4 with a different keyword.
    """,
    functions=[get_wikipedia_summary, search_wikipedia]
)


# %% run the swarm
client = Swarm()
messages = [
    {"role": "user",
     "content": "What is swarm intelligence?"}
]

response = client.run(agent= wikipedia_agent,
                      messages=messages)

#%% check the output
response.model_dump()


# %% TEST
wikipedia.search("Swarm Intelligence")
# %% agent response
pprint(response.messages[-1]['content'])
