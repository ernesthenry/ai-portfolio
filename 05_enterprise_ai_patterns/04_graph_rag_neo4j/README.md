# Problem 3: GraphRAG

**JD Responsibility:** "Knowledge graph integration... Neo4j."

**The Challenge:** Vector search fails to retrieve "multi-hop" relationships (e.g., A knows B, B knows C -> Does A know C?).

**The Solution:**

1. **Neo4j + LlamaIndex:** Extracted entities and edges from raw text.
2. **Graph Querying:** The system traverses the graph to find connected nodes before answering, providing deeper narrative analytics.
