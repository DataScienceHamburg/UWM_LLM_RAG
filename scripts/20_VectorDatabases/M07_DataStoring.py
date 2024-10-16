#%% Packages
import os
from langchain.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# %% Create instance of embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

