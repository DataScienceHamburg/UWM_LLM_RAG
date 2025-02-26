#%% packages
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import WikipediaLoader
from pprint import pprint
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv(usecwd=True))
#%% load llm
# generator_llm = ChatGroq(model="llama-3.1-8b-instant")
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

#%% load data
loader = WikipediaLoader(query="Albert Einstein", load_max_docs=10)
docs = loader.load()

# %% check the data
pprint(docs[0].page_content)
# %% Generate Testset
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)

# %% analyze the dataset
dataset.to_pandas()

#%% login at app.ragas.ai
# create app token online,
# add it to .env
dataset.upload()

# %% store in local FAISS db
# if folder does not exist, create it
db_path = "rag_eval_db2"
if not os.path.exists(db_path):
    os.makedirs(db_path)
    vector_store = FAISS.from_documents(docs, generator_embeddings)
    
    vector_store.save_local(db_path)
else:
    
    vector_store = FAISS.load_local(db_path, generator_embeddings, allow_dangerous_deserialization=True)
# %% Retrieval
def retrieval(query, db_path):
    
    vector_store=FAISS.load_local(db_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # Ranking the chunks in descending order of similarity
    docs = vector_store.similarity_search(query)
    # Selecting first chunk as the retrieved information

    return docs
#%% Augmentation
def augmentation(query, db_path):
 
    retrieved_context=retrieval(query,db_path)
    retrieved_context_str = "\n".join([doc.page_content for doc in retrieved_context])
    # Creating the prompt
    augmented_prompt=f"""
    You will be given a question and a context.
    Question: {query} 
    Context : {retrieved_context_str}

    Answer purely based on the context provided. Do not use any other information. 

    If you cannot answer the question  based on the provided context, say that you don’t know.
    """
    return retrieved_context, str(augmented_prompt)
# %%
def create_rag(user_query, db_path):

    augmented_prompt, retrieved_context=augmentation(user_query,db_path)
    model = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template(augmented_prompt)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    res = chain.invoke({"input": augmented_prompt})

    return retrieved_context, res
# %% TEST RAG Pipeline
user_query = "What is AI?"

#%% 
import numpy as np

class RAG:
    def __init__(self, model="gpt-4o"):
        self.llm = ChatOpenAI(model=model)
        self.embeddings = OpenAIEmbeddings()
        self.doc_embeddings = None
        self.docs = None

    def load_documents(self, documents):
        """Load documents and compute their embeddings."""
        self.docs = documents
        self.doc_embeddings = self.embeddings.embed_documents(documents)

    def get_most_relevant_docs(self, query):
        """Find the most relevant document for a given query."""
        if not self.docs or not self.doc_embeddings:
            raise ValueError("Documents and their embeddings are not loaded.")

        query_embedding = self.embeddings.embed_query(query)
        similarities = [
            np.dot(query_embedding, doc_emb)
            / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            for doc_emb in self.doc_embeddings
        ]
        most_relevant_doc_index = np.argmax(similarities)
        return [self.docs[most_relevant_doc_index]]

    def generate_answer(self, query, relevant_doc):
        """Generate an answer for a given query based on the most relevant document."""
        prompt = f"question: {query}\n\nDocuments: {relevant_doc}"
        messages = [
            ("system", "You are a helpful assistant that answers questions based on given documents only."),
            ("human", prompt),
        ]
        ai_msg = self.llm.invoke(messages)
        return ai_msg.content
# %%
sample_docs = [doc.page_content for doc in docs]

# Initialize RAG instance
rag = RAG()

# Load documents
rag.load_documents(sample_docs)

# Query and retrieve the most relevant document
query = "Who is Albert Einstein?"
relevant_doc = rag.get_most_relevant_docs(query)

# Generate an answer
answer = rag.generate_answer(query, relevant_doc)

print(f"Query: {query}")
print(f"Relevant Document: {relevant_doc}")
print(f"Answer: {answer}")

# %%
sample_questions = dataset.to_pandas().user_input.to_list()
sample_answers = dataset.to_pandas().reference.to_list()

#%%
dataset = []

for query,reference in zip(sample_questions,sample_answers):

    relevant_docs = rag.get_most_relevant_docs(query)
    response = rag.generate_answer(query, relevant_docs)
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":relevant_docs,
            "response":response,
            "reference":reference
        }
    )
# %% create evaluation dataset
from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(dataset)
# %%
llm = ChatOpenAI(model="gpt-4o-mini")


evaluator_llm = LangchainLLMWrapper(llm)


result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy()],llm=evaluator_llm)
# %%
pprint(result)
# %%
result.upload()