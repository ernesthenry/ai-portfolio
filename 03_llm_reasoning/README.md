# LLM Reasoning Engine

This project builds "System 2" thinking (deliberate reasoning) for LLMs. It moves beyond simple text generation to implement structured thought processes.

## Algorithms Implemented

1. **Chain of Thought (CoT)**: Forcing the model to generate a rationale before the answer.
   - _Logic_: $ P(\text{Answer} | \text{Question} + \text{Rationale}) $
2. **Self-Correction**: A critique loop where the model reviews its own output for logical errors.
3. **Tree of Thoughts**: A simplified Breadth-First Search (BFS) exploring multiple reasoning paths.

## Usage

```bash
python solve.py
```

This will run a logic puzzle ("A farmer has 17 sheep...") through all three strategies to demonstrate the difference in reasoning quality.
