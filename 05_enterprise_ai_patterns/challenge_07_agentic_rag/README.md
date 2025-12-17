# Challenge 7: Agentic RAG

**The Challenge:** Queries often require multiple steps (lookup -> calculate -> reason) which single-shot RAG cannot handle.
**The Solution:** Implemented a LangChain Agent with custom tools.

1. **Routing:** The model decides which tool to call.
2. **Multi-step Reasoning:** It takes the output of one tool and feeds it into the next.
