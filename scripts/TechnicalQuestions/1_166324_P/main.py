#%%
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#%%
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#%% Example usage
document = ["Dogs like to go for walks.",
            "They like to run around the park and play with other dogs.",
            "Iwo likes pancakes."]

query = "What dogs like to do?"

#%% get embeddings
document_embeddings = embeddings.embed_documents(document)
query_embedding = embeddings.embed_query(query)

#%% get similarity scores
similarity_scores = cosine_similarity([query_embedding], document_embeddings)

#%% print similarity scores
print("Similarity Scores:")
print(similarity_scores)

# %%
