#%% packages
import os
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv,find_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv(find_dotenv(usecwd=True))
from pprint import pprint
from langchain_chroma import Chroma

#%% data loading
file_path = "../data/data_source.docx"
loader = Docx2txtLoader(file_path)
docs_word = loader.load()


# %% visual inspection
pprint(docs_word[0].page_content[:400])

#%% text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["##","\n\n", "\n", " ", ""],
    chunk_size=1000, 
    chunk_overlap=200
)
docs_split = text_splitter.split_documents(docs_word)

# %% find the number of chunks
len(docs_split)
# %%
embedding_function = OpenAIEmbeddings()
#%%
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_db_path = os.path.join(current_dir, "db")

db = Chroma(persist_directory=persistent_db_path, embedding_function=embedding_function)


# %%
db.add_documents(docs_split)
# %%
len(db.get()['ids'])


# %% set up a retriever
retriever = db.as_retriever()
# %%
retriever.invoke("Czy istnieje specjalna dieta podczas radioterapii i jaki ma charakter?")

# %%
