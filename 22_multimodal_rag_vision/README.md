# Multimodal RAG (Vision)



**The Problem:**
70% of enterprise knowledge is locked in **PDFs, Charts, and Screenshots**, which standard RAG (Text-only) ignores.

**The Solution:**

1.  **Image-to-Text:** Use GPT-4o / LLaVA to caption every image during indexing.
2.  **Indexing:** Store the caption in the Vector DB (Hybrid Search).
3.  **Generation:** When retrieved, pass the _Original Image_ to the VLM to answer specific questions (e.g., "What is the trend in this bar graph?").
