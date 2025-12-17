# Problem 4: Model Routing & Cost Optimization



**The Challenge:** The system was routing simple "Hello" queries to GPT-4, wasting budget.

**The Solution:**

1. **Semantic Router:** Implemented an embedding-based classifier layer.
2. **Logic:**
   - Simple queries -> Static response / Llama-3 (Cheap).
   - Complex reasoning -> GPT-4 (Expensive).
   - **Result:** Estimated 40-60% cost reduction.
