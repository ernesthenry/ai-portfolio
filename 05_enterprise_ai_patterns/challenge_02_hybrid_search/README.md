# Problem 2: Hybrid Search & Evaluation



**The Challenge:** Users complained that specific queries (names, dates) returned irrelevant results despite high semantic similarity.

**The Solution:**

1. **Hybrid Retrieval:** Combined BM25 (Keyword) and Chroma (Vector) using `EnsembleRetriever`.
2. **Evaluation:** Setup structure for RAGAS to score retrieval quality (Context Precision).
