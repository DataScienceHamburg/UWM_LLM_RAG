#%% packages
import networkx as nx
from transformers import pipeline
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from pprint import pprint
#%% create graph
# Create a simple knowledge graph using NetworkX
G = nx.Graph()

# Add nodes representing entities and relationships
G.add_node("Car", type="object", description="A car is a road vehicle typically powered by an engine, used for transporting passengers.")
G.add_node("Engine", type="object", description="The machine that provides power to move the car.")
G.add_node("Fuel", type="object", description="A substance that cars burn to create energy. Examples include gasoline, diesel, and electricity.")
G.add_node("Manufacturer", type="object", description="A company that produces cars. Examples include Ford, Toyota, BMW.")
G.add_node("Wheels", type="object", description="Circular objects that allow the car to move.")
G.add_node("Transmission", type="object", description="The system that transmits power from the engine to the wheels.")
G.add_node("Brakes", type="object", description="Components that slow down or stop the car.")
G.add_node("Battery", type="object", description="Provides electrical energy to start the engine and power electrical systems.")
G.add_node("Steering Wheel", type="object", description="Control mechanism that allows the driver to steer the car.")
G.add_node("Tires", type="object", description="Rubber coverings placed on wheels that provide traction.")
G.add_node("Suspension", type="object", description="System of springs and shock absorbers connecting wheels to the car body.")
G.add_node("Exhaust System", type="object", description="Removes waste gases from the engine and reduces noise.")
G.add_node("Air Conditioning", type="object", description="System that cools and dehumidifies the air inside the car.")
G.add_node("Airbags", type="object", description="Safety devices that inflate during collisions to protect occupants.")
G.add_node("Headlights", type="object", description="Lights at the front of the car that illuminate the road ahead.")
G.add_node("Dashboard", type="object", description="Panel that houses instruments and controls for the driver.")
G.add_node("Seats", type="object", description="Furniture within the car where passengers sit.")
G.add_node("Windshield", type="object", description="The front window of the car that protects occupants from wind and debris.")
G.add_node("Fuel Tank", type="object", description="Container that stores fuel for the engine.")
G.add_node("GPS Navigation", type="object", description="System that provides directions and location information.")
G.add_node("Driver", type="person", description="Person who operates the car.")

# Add relationships (edges) between the entities
G.add_edge("Car", "Engine", relationship="Has")
G.add_edge("Car", "Fuel", relationship="Uses")
G.add_edge("Car", "Manufacturer", relationship="Made by")
G.add_edge("Car", "Wheels", relationship="Has")
G.add_edge("Car", "Transmission", relationship="Contains")
G.add_edge("Car", "Brakes", relationship="Has")
G.add_edge("Car", "Battery", relationship="Contains")
G.add_edge("Car", "Steering Wheel", relationship="Has")
G.add_edge("Car", "Tires", relationship="Uses")
G.add_edge("Car", "Suspension", relationship="Has")
G.add_edge("Car", "Exhaust System", relationship="Contains")
G.add_edge("Car", "Air Conditioning", relationship="Features")
G.add_edge("Car", "Airbags", relationship="Includes")
G.add_edge("Car", "Headlights", relationship="Has")
G.add_edge("Car", "Dashboard", relationship="Contains")
G.add_edge("Car", "Seats", relationship="Has")
G.add_edge("Car", "Windshield", relationship="Features")
G.add_edge("Car", "Fuel Tank", relationship="Contains")
G.add_edge("Car", "GPS Navigation", relationship="May include")
G.add_edge("Driver", "Car", relationship="Operates")
G.add_edge("Wheels", "Tires", relationship="Fitted with")
G.add_edge("Engine", "Fuel", relationship="Consumes")
G.add_edge("Engine", "Battery", relationship="Started by")
G.add_edge("Engine", "Exhaust System", relationship="Connected to")
G.add_edge("Fuel", "Fuel Tank", relationship="Stored in")


#%% Nodes
G.nodes

#%% Edges
G.edges


#%% visualize graph
nx.draw(G, with_labels=True)

#%% graph RAG
# Function to retrieve relevant nodes based on a query
def retrieve_relevant_nodes(query, graph=G):
    """
    Retrieve relevant nodes based on a query using a graph database.
    
    Args:
        query (str): The user's query
        graph (nx.Graph): The knowledge graph to search in

    Returns:
        list: A list of relevant nodes and their descriptions
    """
    # Initialize a list to store relevant nodes
    relevant_nodes = []
    
    # Create an embedding model for semantic similarity
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", tokenizer=tokenizer)
    
    # Get embeddings for the query
    query_embedding = model(query)[0][0]
    
    # Calculate similarity between query and each node's description
    for node, data in graph.nodes(data=True):
        if 'description' in data:
            # Get embedding for node description
            node_embedding = model(data['description'])[0][0]
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Reshape embeddings for sklearn's cosine_similarity function
            query_embedding_reshaped = np.array(query_embedding).reshape(1, -1)
            node_embedding_reshaped = np.array(node_embedding).reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embedding_reshaped, node_embedding_reshaped)[0][0]
            # Add node to relevant nodes if similarity is above threshold
            
            if similarity > 0.5:  # Adjust threshold as needed
                if len(relevant_nodes) >= 3:
                    break
                relevant_nodes.append({
                    'node': node,
                    'description': data.get('description', ''),
                    'similarity': similarity
                })
            # max 3 relevant nodes
            
    # Sort by similarity score (highest first)
    relevant_nodes.sort(key=lambda x: x['similarity'], reverse=True)
    
    return relevant_nodes

#%% create generation function based on LLM based on relevant nodes
def generate_response(query, graph=G):
    relevant_nodes = retrieve_relevant_nodes(query, graph)
    # relevant nodes is a list of dictionaries with node, description, and similarity
    # we want to create a prompt that uses the description of the most relevant nodes to answer the query
    relevant_nodes_str = "\n".join([f"{node['node']}: {node['description']}" for node in relevant_nodes])
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answer questions about the car and its components."),
        ("user", "Answer the following question <<{query}>> based on the following information: {relevant_nodes_str}")
    ])
    
    llm = ChatGroq(model_name="llama3-8b-8192")
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"query": query, "relevant_nodes_str": relevant_nodes_str})
    
    return response

# %%
user_query = "What is the relationship between the car, wheels, and the tires?"
pprint(generate_response(user_query))

# %%
