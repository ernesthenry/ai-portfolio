class ReasoningEngine:
    def __init__(self, llm):
        self.llm = llm

    # STRATEGY 1: Chain of Thought (CoT)
    # The classic "Let's think step by step" approach.
    def chain_of_thought(self, question):
        template = f"""
        Question: {question}
        Answer: Let's think step by step.
        """

        print("--- [CoT] Generating Rationale ---")
        rationale = self.llm.generate(template, max_new_tokens=512)

        # Now force the final answer extraction
        final_prompt = f"{template}{rationale}\nTherefore, the final answer is:"
        answer = self.llm.generate(final_prompt, max_new_tokens=50)

        return rationale, answer


    # STRATEGY 2: Self-Correction (The Critic)
    # 1. Draft -> 2. Critique -> 3. Refine
    def self_correction(self, question):
        # 1. Draft
        draft = self.llm.generate(f"Question: {question}\nAnswer:")
        print(f"--- [Draft] {draft} ---")

        # 2. Critique
        critique_prompt = f"""
        Question: {question}
        Proposed Answer: {draft}

        Review the proposed answer for logical errors. If it is correct, say 'CORRECT'. 
        If it is wrong, explain why.
        Critique:
        """
        critique = self.llm.generate(critique_prompt)
        print(f"--- [Critique] {critique} ---")

        if "CORRECT" in critique.upper() and len(critique) < 20:
            return draft

        # 3. Refine
        fix_prompt = f"""
        Question: {question}
        Proposed Answer: {draft}
        Critique: {critique}

        Based on the critique, provide the corrected final answer.
        Corrected Answer:
        """
        final_answer = self.llm.generate(fix_prompt)
        return final_answer


    # STRATEGY 3: Tree of Thoughts (Simplified BFS)
    # Explores multiple reasoning paths and prunes bad ones.
    def tree_of_thoughts(self, question, breadth=3, depth=3):
        current_thoughts = [f"Question: {question}\nStep 1:"]

        for step_i in range(depth):
            print(f"--- [Tree] Depth {step_i + 1}, Candidates: {len(current_thoughts)} ---")
            new_thoughts = []

            # 1. Expand (Branching)
            for thought_path in current_thoughts:
                for _ in range(breadth):
                    # Generate a short next step
                    continuation = self.llm.generate(thought_path, max_new_tokens=64)
                    new_path = thought_path + " " + continuation
                    new_thoughts.append(new_path)

            # 2. Evaluate (Pruning)
            scored_thoughts = []
            for path in new_thoughts:
                # Heuristic: Ask model if this path is promising (Yes/No)
                # Or use get_score similar to perplexity
                eval_prompt = f"{path}\n\nIs the reasoning above logical and leading to a correct solution? (Yes/No)"
                eval_out = self.llm.generate(eval_prompt, max_new_tokens=5)

                score = 1.0 if "Yes" in eval_out else 0.0
                scored_thoughts.append((score, path))

            # Keep top k (Beam Search)
            scored_thoughts.sort(key=lambda x: x[0], reverse=True)
            current_thoughts = [x[1] for x in scored_thoughts[:breadth]]

        return current_thoughts[0]
