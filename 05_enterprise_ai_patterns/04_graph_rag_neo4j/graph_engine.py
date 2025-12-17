import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core import StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.openai import OpenAI

# NOTE: Requires a running Neo4j instance
# docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

def build_knowledge_graph():
    # 1. CONNECT TO NEO4J
    graph_store = Neo4jGraphStore(
        username="neo4j",
        password="password",
        url="bolt://localhost:7687",
        database="neo4j",
    )
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    # 2. LOAD DATA
    # Create a dummy file
    with open("narrative.txt", "w") as f:
        f.write("Elon Musk works with Linda Yaccarino at X. Linda focuses on advertising.")
    
    documents = SimpleDirectoryReader(".").load_data()

    # 3. EXTRACT TRIPLETS (The 'Graph' Magic)
    # The LLM will parse: (Elon Musk) --[WORKS_WITH]--> (Linda Yaccarino)
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    
    print("Extracting entities and relationships (this takes time)...")
    index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=2,
        llm=llm,
        include_embeddings=True # Hybrid: Graph + Vector
    )

    # 4. QUERY
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the relationship between Elon and Linda?")
    print(f"\nAnswer: {response}")

if __name__ == "__main__":
    # Wrap in try-except in case Neo4j isn't running during interview demo
    try:
        build_knowledge_graph()
    except Exception as e:
        print("Neo4j not detected. Please run Docker container.")
        print(e)
