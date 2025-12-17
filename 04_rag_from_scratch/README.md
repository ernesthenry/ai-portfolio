# RAG From Scratch (No LangChain)

This project builds a Retrieval-Augmented Generation (RAG) system using pure NumPy and PyTorch. It demonstrates the mathematics of vector search without relying on abstraction frameworks like LangChain or Vector Databases like Chroma/Pinecone.

## Core Concepts

1. **Chunking**: Sliding window approach to split text.
2. **Embeddings**: Using `sentence-transformers` to convert text to vectors.
3. **Vector Search**: Implementing `Cosine Similarity` using Matrix Multiplication ($ A \cdot B^T $).
4. **Context Injection**: Augmenting the LLM prompt with retrieved chunks.

## Usage

1. Ensure `knowledge_base.txt` has data.
2. Run the application:
   ```bash
   python main.py
   ```

## Why Build From Scratch?

Understanding the underlying matrix operations of vector search is critical for optimizing retrieval latency and quality in production systems.
