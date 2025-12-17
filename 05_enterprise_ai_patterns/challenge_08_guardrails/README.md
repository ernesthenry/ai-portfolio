# Challenge 8: Enterprise Guardrails

**The Challenge:** Large enterprises (CoreStory's clients) cannot risk Prompt Injection or PII leakage.
**The Solution:**

1. **Input Guard:** Keyword/Semantic analysis before hitting the LLM (saves cost + ensures safety).
2. **Output Guard:** Regex/Presidio scrubbing to ensure no private data leaves the system.
