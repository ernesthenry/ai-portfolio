# Challenge 10: Human-in-the-Loop (Persistence)

**The Challenge:** Enterprise agents cannot run fully autonomously on sensitive tasks.
**The Solution:** Implemented **LangGraph Checkpointing**.

1. Agent drafts content.
2. System **pauses** execution and saves memory.
3. Human reviews and modifies state (approves).
4. System resumes execution.
