#%% packages
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
#%% load the db
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_db_path = os.path.join(current_dir, "dbnew")
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=persistent_db_path, embedding_function=embedding_function, collection_name="advanced_rag")
retriever = db.as_retriever()
# %% get all docs
# db.get()

#%%
def rag(user_query:str, language = "German", n_results=5):
    docs = retriever.invoke(user_query)[:n_results]
    print(docs)
    docs_content = [doc.page_content for doc in docs]
    joined_information = ';'.join([f'{doc.page_content}' for doc in docs])
    print(f"joined information: {joined_information}")
    messages = [
        ("system", "You are a historian. Your users are asking questions about information contained in attached information. You will be shown the user's question, and the relevant information. Answer the user's question using only this information. Say 'I don't know' if you don't know the answer. Answer in specified language."),
        ("user", f"Question: {user_query}. \n Information: {joined_information}. \n Language: {language}")
    ]
    print(f"messages: {messages}")
    prompt = ChatPromptTemplate.from_messages(messages)
    model = ChatOpenAI()
    chain = prompt | model | StrOutputParser()
    res = chain.invoke({})
    # return also the complete prompt
    return docs_content, res, prompt.invoke({"query": user_query, "joined_information": joined_information, "language": language})

# %% Test
# raw_docs, rag_response, prompt = rag(query="Ist eine bestimmte Diät während der Radiotherapie erforderlich?", language="German", n_results=3)
# %% extract only the docs page content
st.header("History Expert on Industrial Revolution")

# text input field
user_query = st.text_input(label="User Query", help="Raise your questions You can ask in any language.", placeholder="What do you want to know?")

#%% run query only after button is pressed
raw_docs = ["", "", "", "", ""]
rag_response = ""
prompt = None
#%% add a dropdown for language
language = st.selectbox("Output Language", ["German", "English", "Polish"])

#%% run query only after button is pressed
if st.button("Ask"):
    raw_docs, rag_response, prompt = rag(user_query, language=language, n_results=3)
    print(f"raw docs: {raw_docs}")
    print(f"rag response: {rag_response}")

st.header("Retrieval")
st.markdown(f"**Raw Response 0:** {raw_docs[0]}")
st.markdown(f"**Raw Response 1:** {raw_docs[1]}")
st.markdown(f"**Raw Response 2:** {raw_docs[2]}")
st.header("Augmentation")
if prompt:
    st.markdown(prompt.messages[1].content)

st.header("Generation")
st.write(rag_response)

#%%

