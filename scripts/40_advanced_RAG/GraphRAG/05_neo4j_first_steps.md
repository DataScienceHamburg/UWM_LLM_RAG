# Check server connection

```neo4j
:server connect
```

# Create Nodes

```neo4j
MERGE (p:Person {id: 'bob', name: 'Bob'})
MERGE (p:Person {id: 'alice', name: 'Alice'})
MERGE (p:Person {id: 'eve', name: 'Eve'})
```

# Check the nodes

```neo4j
MATCH (p:Person) RETURN p
```

# Create Relationships

```neo4j
MATCH (a:Person {id: 'bob'}), (b:Person {id: 'eve'}) MERGE (b)-[:COLLEAGUE]-(a);
```

# Query the node parameters

```neo4j
MATCH (n)-[r]->(m) RETURN n,r,m
```
