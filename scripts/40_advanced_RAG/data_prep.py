#%% packages
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
import os
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
# %%
topic = "Industrial Revolution"
loader = WikipediaLoader(query=topic, load_max_docs=1, doc_content_chars_max=8000)
docs = loader.load()

#%% Recursive Character Text chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
chunks_recursive = splitter.split_documents(docs)
len(chunks_recursive)

#%% check the chunk
pprint(chunks_recursive[1].page_content)

#%%
pprint(chunks_recursive[2].page_content)

#%% semantic chunking
# TODO: implement semantic chunking
splitter = SemanticChunker(embeddings=OpenAIEmbeddings(), 
                           number_of_chunks=20)
chunks_semantic = splitter.split_documents(docs)
#%%
len(chunks_semantic)

#%% check the chunk
pprint(chunks_semantic[1].page_content)

#%%
pprint(chunks_semantic[2].page_content)

# %% create chunks
# contextual retriever should have small chunks with not more than a few hundred tokens
# source: https://www.anthropic.com/news/contextual-retrieval
# e.g. 200 tokens is about 800 characters

#%% create function for context information
def create_chunks_with_context(document: Document):
    model = ChatGroq(model_name="llama-3.2-3b-preview")

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
         You are part of a retrieval augmented generation pipeline. You are given a complete document, and a chunk of the document.
         Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. Don't repeat the chunk content.
         """),
        ("user", "<document>{document}</document><chunk>{chunk}</chunk>"),
    ])
    chain = prompt | model | StrOutputParser()

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitter = SemanticChunker(embeddings=OpenAIEmbeddings(), 
                           number_of_chunks=20)
    chunks = splitter.split_documents([document])
    context_chunks = []
    for i, chunk in enumerate(chunks):
        print(i)
        response = chain.invoke({"document": document.page_content, "chunk": chunk})
        context_chunks.append(";".join([response, chunk.page_content]))

    return context_chunks

#%% check the number of chunks
context_chunks = create_chunks_with_context(docs[0])


#%% 
pprint(context_chunks[5])
# %%
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_db_path = os.path.join(current_dir, "dbnew")

embedding_function = OpenAIEmbeddings()

db = Chroma(persist_directory=persistent_db_path, 
            collection_name="advanced_rag", 
            embedding_function=embedding_function)


# %%
db.add_texts(context_chunks)
# %%
len(db.get()['ids'])


# %% set up a retriever
retriever = db.as_retriever()
# %%
retriever.invoke("What happened in the Industrial Revolution?")


# %%
db.get()

# %%
