from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# DUMMY DATA: Note how vector search might confuse these
docs = [
    Document(page_content="The Apple iPhone 15 was released in 2023.", metadata={"year": 2023}),
    Document(page_content="Apple released the iPhone 4 in 2010.", metadata={"year": 2010}),
    Document(page_content="Apples are a fruit grown in orchards.", metadata={"topic": "fruit"}),
]

def run_hybrid_search():
    embedding = OpenAIEmbeddings()

    # 1. SPARSE RETRIEVER (BM25 - Keyword Match)
    # Good for exact matches like "2010" or specific names
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2

    # 2. DENSE RETRIEVER (Vector - Semantic Match)
    # Good for understanding "mobile phone devices" matches "iPhone"
    vectorstore = Chroma.from_documents(docs, embedding)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. ENSEMBLE (The Hybrid Logic)
    # We weight BM25 50% and Vector 50%
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )

    # 4. TEST
    query = "Apple phone from 2010"
    print(f"Querying: {query}")
    results = ensemble_retriever.invoke(query)

    for doc in results:
        print(f"- {doc.page_content}")

    # --- EVALUATION MOCKUP (RAGAS) ---
    # In a real run, we would pass these retrieved contexts to Ragas
    # to calculate 'Context Precision'.
    print("\n[Metric Log] Context Recall: 1.0 (Retrieved correct doc)")
    print("[Metric Log] Faithfulness: 1.0 (Answer derived from doc)")

if __name__ == "__main__":
    run_hybrid_search()
