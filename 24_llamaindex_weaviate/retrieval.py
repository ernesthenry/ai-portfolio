import weaviate
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import os

# CONCEPT: Scale-out Vector Search
# Chroma is great for local. Weaviate/Pinecone is for Production (100M+ Vectors).

def ingest_to_weaviate():
    print("--- Connecting to Weaviate (Cluster) ---")
    
    # 1. CONNECT TO VECTOR DB
    # In prod, this URL points to your Cloud Cluster
    client = weaviate.Client("http://localhost:8080")
    
    # 2. DEFINE SCHEMA (LlamaIndex does this, but good to know)
    # Weaviate uses classes (like SQL tables)
    
    # 3. SETUP LLAMAINDEX with WEAVIATE
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name="EnterpriseDocs")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 4. LOAD & CHUNK
    # "Recursive Retrieval" is a LlamaIndex specialty (Small chunk to retrieve, Big chunk to generate)
    documents = SimpleDirectoryReader("./data").load_data()
    
    # 5. INDEX
    print("Indexing documents into Weaviate...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    
    # 6. QUERY
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the refund policy?")
    print(f"Response: {response}")

if __name__ == "__main__":
    # Mock run if no Weaviate instance
    try:
        ingest_to_weaviate()
    except Exception as e:
        print("Weaviate not running. This code demonstrates the integration pattern.")
        print("Pattern: Documents -> LlamaIndex NodeParser -> Weaviate Vector Store -> Retrieval")
