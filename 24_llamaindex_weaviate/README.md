# Enterprise Retrieval (LlamaIndex + Weaviate)



**The Business Problem:**
ChromaDB crashes when you have 10 Million PDF pages. You need a Distributed Vector Database.

**The Solution:**

1.  **Orchestrator:** LlamaIndex (Best-in-class for data ingestion/parsing).
2.  **Storage:** Weaviate (Cloud-native, scalable vector engine).
3.  **Pattern:** **Recursive Retrieval**. We index "Small Chunks" (sentences) for high-precision search, but retrieve the "Parent Chunk" (Paragraph) to give the LLM enough context.
