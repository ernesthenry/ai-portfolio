from llm_wrapper import LLMWrapper
from algorithms import ReasoningEngine

def main():
    # 1. Initialize
    print("Initializing Reasoning Engine...")
    try:
        llm = LLMWrapper(device="cuda") # Will auto-fallback to cpu in wrapper if needed
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    engine = ReasoningEngine(llm)
    
    # A tricky logic puzzle
    problem = """
    A farmer has 17 sheep. All but 9 die. How many sheep are left?
    """
    
    print(f"PROBLEM: {problem}\n")
    
    # --- Run Strategy 1: CoT ---
    print(">>> STRATEGY 1: CHAIN OF THOUGHT")
    rationale, answer = engine.chain_of_thought(problem)
    print(f"Rationale: {rationale}")
    print(f"Final Answer: {answer}\n")
    
    # --- Run Strategy 2: Self-Correction ---
    print(">>> STRATEGY 2: SELF-CORRECTION")
    final = engine.self_correction(problem)
    print(f"Final Answer: {final}\n")
    
    # --- Run Strategy 3: Tree of Thoughts ---
    print(">>> STRATEGY 3: TREE OF THOUGHTS")
    best_path = engine.tree_of_thoughts(problem, breadth=2, depth=2)
    print(f"Best Reasoning Path: {best_path}")

if __name__ == "__main__":
    main()
