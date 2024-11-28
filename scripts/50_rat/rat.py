#%% packages
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import WikipediaLoader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from typing import List
from langchain_core.output_parsers import JsonOutputParser

#%% model setup
embedding = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

# %%
loader = WikipediaLoader(query="Principle of relativity", load_max_docs=10)
docs = loader.load()
# %%
docs
# %%
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
#%% vector DB
vectorstore = Chroma.from_documents(chunks, embedding=embedding)

# %% define a retriever to get the relevant docs
retriever = vectorstore.as_retriever()

# %% test it
# retriever.invoke(input="Who developed the special relativity?")

# %% define the prompt to ask questions and invoke thought process
thought_prompt = PromptTemplate(
    input_variables=["task_prompt", "context", "previous_thoughts"],
    template="""
    You are a knowledgeable assistant using Retrieval-Analyze-Respond (RAT) framework to solve a given task. Make sure your output is a valid JSON object with the following field: thoughts: List[str], answer: str. Formulate each thought as a single question.
    
    <<task_prompt>>{task_prompt}<</task_prompt>>
    
    <<context>>{context}<</context>>
    
    <<previous_thoughts>>{previous_thoughts}<</previous_thoughts>>
    
    Retrieve:
    1. Identify the key information needed to address the task prompt. 
    2. Gather relevant information from the provided context.
    
    Analyze:
    1. Critically evaluate the information gathered.
    2. Consider multiple perspectives and possible solutions.
    3. Reason through the problem to determine the best next step.
    
    Respond:
    Next thought: [RESULT]
    Answer: [RESULT]
    """
)

class StructuredThoughtsAndAnswer(BaseModel):
    thoughts: List[str]
    answer: str

rat_chain = thought_prompt | llm | JsonOutputParser(pydantic_object=StructuredThoughtsAndAnswer)

#%% initial thought prompt
initial_thought_prompt = PromptTemplate(
    input_variables=["task_prompt"],
    template="""
    You are a knowledgeable assistant. Make sure your output is a valid JSON object with the following fields: thoughts = [thought1, thought2, ...]. Formulate each thought as a single question.

    Analyze:
    1. Critically evaluate the information gathered.
    2. Consider multiple perspectives and possible solutions.
    3. Reason through the problem to determine the best next step.

    Respond:
    Next thought: [RESULT]

    Task Prompt: {task_prompt}
    """
)

#%% class for structured output
class StructuredThoughts(BaseModel):
    thoughts: List[str]
    
initial_cot_chain = initial_thought_prompt | llm | JsonOutputParser(pydantic_object=StructuredThoughts)
# %%
task_prompt = "What is relativity? Who developed it? Why was the developer considered a genius? What was his dispute with Mach?"
initial_cot_response = initial_cot_chain.invoke({"task_prompt": task_prompt})
initial_cot_response
# %%
thoughts = initial_cot_response['thoughts']
# %%
for i in range(len(thoughts)):
    print(i)
    # get all previous context
    
    previous_context = "\n".join(thoughts[:i])
    
    # retrieve relevant docs
    relevant_docs = retriever.invoke(thoughts[i])
    
    # combine the context
    relevant_context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # get relevant docs
    query = f"{thoughts[i]}; Context: {relevant_context}"
    print(f"Thought {i+1}: {thoughts[i]}")
    rat_response = rat_chain.invoke({"task_prompt": query, "context": relevant_context, "previous_thoughts": ";".join(thoughts[:i])})
    thoughts[i] = rat_response['answer']
    print(thoughts)
    print("------------------------\n")
# %%
thoughts
# %%
i =0

# %%
