# Problem 5: Structured Extraction Agent



**The Challenge:** Need to convert unstructured interviews into database rows without hallucinated formatting.

**The Solution:**

1. **Pydantic + Instructor:** Defined a strict data schema.
2. **Self-Correction:** Implemented validation logic (e.g., sentiment score range). If the LLM generates an invalid number, the code rejects it and asks the LLM to fix it automatically.
