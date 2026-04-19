import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./data/chromadb")
        # Using default embedding function (sentence-transformers)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="notes", 
            embedding_function=self.embedding_fn
        )

    def add_document(self, doc_id: str, text: str, metadata: dict):
        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata]
        )

    def query(self, query_text: str, n_results: int = 3):
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
