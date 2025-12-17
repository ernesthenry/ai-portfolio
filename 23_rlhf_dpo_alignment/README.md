# Post-Training Alignment (RLHF / DPO)

**JD Alignment:** "Post training reinforcement."

**The Business Problem:**
Base LLMs are smart but **unaligned**. They might explain how to build a bomb or be rude to customers.
Standard Fine-Tuning (SFT) teaches facts. **RLHF/DPO** teaches _style, safety, and preference_.

**The Solution:**
Direct Preference Optimization (DPO).

- We feed the model pairs of `(Query, Good_Answer, Bad_Answer)`.
- The model mathematically updates its weights to increase the probability of `Good_Answer` while staying close to the original model (KL Divergence).
