import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class GraphStore:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def add_knowledge(self, kg_data):
        with self.driver.session() as session:
            # Add entities
            for entity in kg_data.entities:
                session.run("MERGE (e:Entity {name: $name})", name=entity)
            
            # Add relations
            for rel in kg_data.relations:
                session.run(
                    "MATCH (a:Entity {name: $source}), (b:Entity {name: $target}) "
                    "MERGE (a)-[r:RELATED {type: $rel_type}]->(b)",
                    source=rel.source,
                    target=rel.target,
                    rel_type=rel.relation
                )

    def query_graph(self, entity_name: str):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {name: $name})-[r]-(connected) RETURN connected.name, type(r)",
                name=entity_name
            )
            return [record.values() for record in result]
