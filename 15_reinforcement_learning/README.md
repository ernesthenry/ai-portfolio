# Reinforcement Learning (Q-Learning)

**Goal:** Demonstrate agents that learn from _environment feedback_ rather than labeled data.

## The Problem (GridWorld)

An agent is dropped in a maze. It gets:

- `+10` points for finding Gold.
- `-10` points for falling in a Pit.
- `-1` point for every second it wastes.

## The Algorithm: Q-Learning

This uses the **Bellman Equation** to discover the "Value" of being in a state.
Unlike LLMs (Sequence Prediction) or Classifiers (Pattern Matching), this is **Decision Making**.
