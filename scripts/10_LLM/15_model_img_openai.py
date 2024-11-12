#%% packages
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

#%%
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="A cute baby sea otter, 8-bit ascii art",
  n=1,
  size="1024x1024"
)

# %%
response.data[0].url
