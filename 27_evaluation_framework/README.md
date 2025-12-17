# AI Evaluation Framework (Ragas / LLM-as-a-Judge)

**The Final Problem:** "How do we know if our AI is good?"

**The Solution:** Automated Evaluation.
Instead of staring at chat logs, we run a pipeline where a stronger model (GPT-4) grades the weaker model (Llama-3).

**Key Metrics:**

1.  **Faithfulness:** Is the answer derived _only_ from the retrieved documents? (Hallucination check).
2.  **Answer Relevance:** Did it actually help the user?
3.  **Context Recall:** Did the RAG system find the right document?

**Why this matters:**
You cannot deploy to Enterprise without a "Test Score". This module provides that score.
