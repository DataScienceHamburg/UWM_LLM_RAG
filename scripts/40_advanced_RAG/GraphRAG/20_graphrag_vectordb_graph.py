#%% packages
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph, Neo4jVector
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain_openai import OpenAIEmbeddings
import os
import numpy as np

#%% connect to the graph
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database="neo4j",  # Specify database name

)

# %% fetch all nodes
graph.query("MATCH (n) RETURN n")

# %% create some nodes
queries = [
    # delete all nodes
    "MATCH (n) DETACH DELETE n",
    # Create 5 Employee nodes with different roles, bios, and availability
    "CREATE (e1:Employee {name: 'Alice', role: 'Software Engineer', bio: 'Experienced in backend development with Python and Django. Also skilled in UX design and has led several successful UI/UX initiatives. Passionate about creating both scalable and user-friendly systems.', currently_available: true})",
    "CREATE (e2:Employee {name: 'Bob', role: 'Data Scientist', bio: 'Strong background in statistical analysis and machine learning. Previously worked as a full-stack developer, bringing valuable software engineering practices to data science projects. Proficient in Python, R, and JavaScript.', currently_available: false})",
    "CREATE (e3:Employee {name: 'Charlie', role: 'UX Designer', bio: 'User-centered design advocate with expertise in Figma and user research. Has a computer science degree and regularly contributes to frontend development using React. Passionate about accessible design.', currently_available: true})",
    "CREATE (e4:Employee {name: 'Diana', role: 'Project Manager', bio: 'Proven track record of leading agile teams. Former software developer who transitioned to management, bringing strong technical understanding to project planning. Certified in both PMP and AWS.', currently_available: true})", 
    "CREATE (e5:Employee {name: 'Eve', role: 'Frontend Developer', bio: 'Specializes in React and JavaScript. Has a background in graphic design and animation, creating engaging micro-interactions. Also experienced in mobile app development with React Native.', currently_available: false})",
    "CREATE (e6:Employee {name: 'Frank', role: 'DevOps Engineer', bio: 'Expert in CI/CD pipelines and cloud infrastructure. Previously worked as a security consultant, bringing strong security practices to DevOps. Skilled in AWS, Docker, and penetration testing.', currently_available: true})",
    "CREATE (e7:Employee {name: 'Grace', role: 'Security Engineer', bio: 'Specialized in application security and penetration testing. CISSP certified. Has extensive experience in Python development and machine learning for threat detection.', currently_available: false})",
    "CREATE (e8:Employee {name: 'Henry', role: 'Mobile Developer', bio: 'Experienced in native iOS and Android development. Also skilled in backend development with Node.js and MongoDB. Previously worked as a game developer, bringing unique UI animation skills.', currently_available: true})",
    "CREATE (e9:Employee {name: 'Ivy', role: 'Database Administrator', bio: 'Expert in database optimization and maintenance. Has significant experience in data science and machine learning, particularly in building ML pipelines. Skilled in SQL, NoSQL, and Python.', currently_available: true})",
    "CREATE (e10:Employee {name: 'Jack', role: 'QA Engineer', bio: 'Focused on automated testing and quality assurance. Previously worked as a full-stack developer and brings strong coding skills to test automation. Expert in testing frameworks and performance optimization.', currently_available: false})",

    # Create 5 Training nodes with different topics
    "CREATE (t1:Training {name: 'Python for Data Science', duration: '3 days'})",
    "CREATE (t2:Training {name: 'Agile Project Management', duration: '2 days'})",
    "CREATE (t3:Training {name: 'Advanced React Concepts', duration: '4 days'})",
    "CREATE (t4:Training {name: 'UX Design Principles', duration: '2 days'})",
    "CREATE (t5:Training {name: 'Cloud Computing Fundamentals', duration: '5 days'})",

    # Create COMPLETED relationships between Employees and Trainings (1-3 trainings per employee)
    "MATCH (e1:Employee {name: 'Alice'}), (t1:Training {name: 'Python for Data Science'}) MERGE (e1)-[:COMPLETED {completion_date: date('2024-11-15')}]->(t1)",
    "MATCH (e2:Employee {name: 'Bob'}), (t1:Training {name: 'Python for Data Science'}) MERGE (e2)-[:COMPLETED {completion_date: date('2025-01-10')}]->(t1)",
    "MATCH (e2:Employee {name: 'Bob'}), (t5:Training {name: 'Cloud Computing Fundamentals'}) MERGE (e2)-[:COMPLETED {completion_date: date('2025-03-01')}]->(t5)",
    "MATCH (e3:Employee {name: 'Charlie'}), (t4:Training {name: 'UX Design Principles'}) MERGE (e3)-[:COMPLETED {completion_date: date('2025-02-18')}]->(t4)",
    "MATCH (e3:Employee {name: 'Charlie'}), (t3:Training {name: 'Advanced React Concepts'}) MERGE (e3)-[:COMPLETED {completion_date: date('2025-01-15')}]->(t3)",
    "MATCH (e4:Employee {name: 'Diana'}), (t2:Training {name: 'Agile Project Management'}) MERGE (e4)-[:COMPLETED {completion_date: date('2024-10-25')}]->(t2)",
    "MATCH (e5:Employee {name: 'Eve'}), (t3:Training {name: 'Advanced React Concepts'}) MERGE (e5)-[:COMPLETED {completion_date: date('2025-04-01')}]->(t3)",
    "MATCH (e6:Employee {name: 'Frank'}), (t5:Training {name: 'Cloud Computing Fundamentals'}) MERGE (e6)-[:COMPLETED {completion_date: date('2024-12-10')}]->(t5)",
    "MATCH (e6:Employee {name: 'Frank'}), (t2:Training {name: 'Agile Project Management'}) MERGE (e6)-[:COMPLETED {completion_date: date('2025-01-20')}]->(t2)",
    "MATCH (e6:Employee {name: 'Frank'}), (t1:Training {name: 'Python for Data Science'}) MERGE (e6)-[:COMPLETED {completion_date: date('2025-02-15')}]->(t1)",
    "MATCH (e7:Employee {name: 'Grace'}), (t1:Training {name: 'Python for Data Science'}) MERGE (e7)-[:COMPLETED {completion_date: date('2024-11-30')}]->(t1)",
    "MATCH (e7:Employee {name: 'Grace'}), (t5:Training {name: 'Cloud Computing Fundamentals'}) MERGE (e7)-[:COMPLETED {completion_date: date('2025-01-05')}]->(t5)",
    "MATCH (e8:Employee {name: 'Henry'}), (t3:Training {name: 'Advanced React Concepts'}) MERGE (e8)-[:COMPLETED {completion_date: date('2024-12-20')}]->(t3)",
    "MATCH (e9:Employee {name: 'Ivy'}), (t1:Training {name: 'Python for Data Science'}) MERGE (e9)-[:COMPLETED {completion_date: date('2025-02-01')}]->(t1)",
    "MATCH (e9:Employee {name: 'Ivy'}), (t2:Training {name: 'Agile Project Management'}) MERGE (e9)-[:COMPLETED {completion_date: date('2025-03-10')}]->(t2)",
    "MATCH (e10:Employee {name: 'Jack'}), (t2:Training {name: 'Agile Project Management'}) MERGE (e10)-[:COMPLETED {completion_date: date('2024-12-15')}]->(t2)",
]

for query in queries:
    graph.query(query)

# %%
# Create an embedding for the "bio" property
# First, create a vector index for the bio field
graph.query("""
CREATE VECTOR INDEX employee_bio_index IF NOT EXISTS
FOR (e:Employee)
ON e.bio
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}}
""")

#%% check the index
graph.query("SHOW INDEXES WHERE type = 'VECTOR'")


#%% Initialize the embeddings model
embeddings = OpenAIEmbeddings()

#%% Fetch all employees with their bios
result = graph.query("MATCH (e:Employee) RETURN e.name AS name, e.bio AS bio, e.role AS role")
employees = [(record["name"], record["bio"], record["role"]) for record in result]
employees
#%% Generate embeddings for each bio and update the nodes
for name, bio, role in employees:
    if bio:
        # Generate embedding for the bio
        embedding = embeddings.embed_query(bio)
        
        # Convert the embedding to a format Neo4j can store
        embedding_array = np.array(embedding, dtype=np.float32).tolist()
        
        # Update the node with the embedding
        graph.query(
            "MATCH (e:Employee {name: $name}) SET e.bio_embedding = $embedding",
            {"name": name, "embedding": embedding_array}
        )

print(f"Created and stored embeddings for {len(employees)} employee bios")

# %% make a query based on embedding
user_query = "a cloud computing expert"
user_query_embedding = embeddings.embed_query(user_query)
user_query_embedding_array = np.array(user_query_embedding, dtype=np.float32).tolist()
top_k = 5


# %%
result = graph.query("""
            MATCH (e:Employee)
            WHERE e.bio_embedding IS NOT NULL 
                     AND e.currently_available = true
                     AND EXISTS((e)-[:COMPLETED]->(:Training {name: "Python for Data Science"}))
            WITH e, gds.similarity.cosine(e.bio_embedding, $query_vector) AS similarity
            RETURN e.name, e.bio, similarity
            ORDER BY similarity DESC
            LIMIT 10
            """,
            params={"query_vector": user_query_embedding_array}
            )
print(result)

# %%
for record in result:
    print(record)
# %%

