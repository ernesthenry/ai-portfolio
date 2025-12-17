# Challenge 9: Multi-Agent Orchestration (LangGraph)

**The Challenge:** Single LLMs struggle with complex, multi-step tasks requiring different "mindsets" (Creative vs Analytical).
**The Solution:** Implemented a StateGraph where:

1. **Researcher** fetches raw info.
2. **Analyst** processes it.
3. **Writer** formats it.
   This "Separation of Concerns" dramatically improves output quality.
