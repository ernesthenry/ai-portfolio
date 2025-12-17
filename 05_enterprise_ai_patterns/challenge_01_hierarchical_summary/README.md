# Problem 1: Hierarchical Summarization

**JD Responsibility:** "Automated summarization and narrative analytics."

**The Challenge:** Summarizing a 100-page document by stuffing it all into context is expensive and degrades performance ("Lost in the Middle").

**The Solution:** Implemented a Map-Reduce chain using LangChain.

1. **Map:** Summarize individual chunks (Chapters).
2. **Reduce:** Summarize the list of chunk summaries.
3. **Result:** A coherent global summary without hitting context limits.
