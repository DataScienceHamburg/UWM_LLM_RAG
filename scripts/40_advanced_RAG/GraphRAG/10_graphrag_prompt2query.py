#%% packages
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser


# %% connect to neo4j
driver = GraphDatabase.driver(
    "neo4j://localhost:7687",
    auth=("neo4j", "test123456")
)

#%% check the connection (no exception means success)
driver.verify_connectivity()

#%%
driver.execute_query("MATCH (n) RETURN n")

#%% create knowledge graph
with driver.session() as session:
    # delete all nodes
    session.run("MATCH (n) DETACH DELETE n")

    # Create the nodes for people
    session.run("MERGE (alice:Person {id: 'alice', name: 'Alice'})")
    session.run("MERGE (bob:Person {id: 'bob', name: 'Bob'})")
    session.run("MERGE (charlie:Person {id: 'charlie', name: 'Charlie'})")
    session.run("MERGE (david:Person {id: 'david', name: 'David'})")

    # Create the nodes for interests
    session.run("MERGE (hiking:Interest {id: 'hiking', name: 'Hiking'})")
    session.run("MERGE (reading:Interest {id: 'reading', name: 'Reading'})")
    session.run("MERGE (photography:Interest {id: 'photography', name: 'Photography'})")
    session.run("MERGE (cooking:Interest {id: 'cooking', name: 'Cooking'})")

    # Create the "likes" relationships
    session.run("MATCH (p:Person {id: 'alice'}), (i:Interest {id: 'hiking'}) MERGE (p)-[:LIKES]->(i)")
    session.run("MATCH (p:Person {id: 'alice'}), (i:Interest {id: 'reading'}) MERGE (p)-[:LIKES]->(i)")
    session.run("MATCH (p:Person {id: 'bob'}), (i:Interest {id: 'photography'}) MERGE (p)-[:LIKES]->(i)")
    session.run("MATCH (p:Person {id: 'bob'}), (i:Interest {id: 'cooking'}) MERGE (p)-[:LIKES]->(i)")
    session.run("MATCH (p:Person {id: 'charlie'}), (i:Interest {id: 'reading'}) MERGE (p)-[:LIKES]->(i)")
    session.run("MATCH (p:Person {id: 'david'}), (i:Interest {id: 'hiking'}) MERGE (p)-[:LIKES]->(i)")
    session.run("MATCH (p:Person {id: 'bob'}), (i:Interest {id: 'hiking'}) MERGE (p)-[:LIKES]->(i)")

    # Create the "is_friend_of" relationships
    session.run("MATCH (charlie:Person {id: 'charlie'}), (alice:Person {id: 'alice'}) MERGE (charlie)-[:IS_FRIEND_OF]-(alice)")
    session.run("MATCH (david:Person {id: 'david'}), (bob:Person {id: 'bob'}) MERGE (david)-[:IS_FRIEND_OF]-(bob)")
    session.run("MATCH (alice:Person {id: 'alice'}), (bob:Person {id: 'bob'}) MERGE (alice)-[:IS_FRIEND_OF]-(bob)")

#%% run some queries
with driver.session() as session:
    result = session.run("RETURN COUNT(*) AS count")
    for record in result:
        print(record["count"])

#%% run a cypher query
with driver.session() as session:
    records, summary, keys = driver.execute_query("MATCH (n) RETURN n")
    print(records[0])

#%% extract all node names
for record in records:
    print(record["n"]["name"])


#%% directly return a dataframe
import pandas as pd
from neo4j import Result, RoutingControl
df = driver.execute_query("MATCH (p:Person) RETURN p.name", result_transformer_=Result.to_df)

df

#%% execute query runs in WRITE mode (default)
driver.execute_query("MATCH (p:Person) RETURN p.name", 
                     result_transformer_=Result.to_df,
                     routing_=RoutingControl.READ)
# %% GraphRAG
def graphRAG(user_query, db_description):
    model = ChatGroq(
        model_name="llama-3.3-70b-versatile") 
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in graph databases, specifically Neo4j and Cypher. For a given db description and a question, you will generate a Cypher query to answer the question. db description: {db_description}. You answer only with the Cypher query, no other text."),
        ("user", "{user_query}"),
        ])
    chain_cypher = prompt_template | model | StrOutputParser()
    response_cypher = chain_cypher.invoke({"db_description": db_description, "user_query": user_query})
    
    with driver.session() as session:
        retrieved_docs = session.run(response_cypher)
        extracted_properties = [record for record in retrieved_docs]

    # extract person names
    person_names = [record.data()["friend"]["name"] for record in extracted_properties]

    # Generate a response
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly assistant. You will be provided with a user_query and an answer. Write a response to the user_query based on the answer."),
        ("user", "user_query: {user_query}\nanswer: {answer}"),
        ])
    chain_response = prompt_template | model | StrOutputParser()
    response = chain_response.invoke({"user_query": user_query, "answer": person_names})
    return response


# %% test
db_description = """
The graph consists of entities: Person, Interest. Relationships: LIKES, IS_FRIEND_OF.
Example: MATCH (charlie:Person {id: 'charlie'}), (alice:Person {id: 'alice'}) MERGE (charlie)-[:IS_FRIEND_OF]-(alice)
MERGE (hiking:Interest {id: 'hiking', name: 'Hiking'})
MERGE (alice:Person {id: 'alice', name: 'Alice'})
"""

user_query = "Who are Alice's friends that also like hiking?"

graphRAG(user_query, db_description)
# %%

# %%
