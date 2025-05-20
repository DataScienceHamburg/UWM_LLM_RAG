#%%
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

#%% Set page config
st.set_page_config(
    page_title="Streaming Chat App",
    page_icon="💬",
    layout="centered"
)

# Initialize Groq chat model
model = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),

)

#%% Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title
st.title("Streaming Chat with Groq")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with streaming effect
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get streaming response from Groq
        for chunk in model.stream(prompt):
            if hasattr(chunk, 'content'):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
