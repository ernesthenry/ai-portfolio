import numpy as np
import time

# REINFORCEMENT LEARNING: Q-Learning
# Problem: An agent needs to navigate a grid to find gold [G] and avoid pits [P].
# [S] .  .
# .   P  .
# .   .  [G]

class GridWorld:
    def __init__(self):
        # 3x3 Grid
        self.state = 0 # Start at 0,0 (index 0)
        self.goal = 8  # Bottom right
        self.pits = [4] # Center
        self.n_states = 9
        self.n_actions = 4 # Up, Down, Left, Right

    def step(self, action):
        # Move logic (simplified)
        row, col = divmod(self.state, 3)
        if action == 0: row = max(0, row-1) # Up
        if action == 1: row = min(2, row+1) # Down
        if action == 2: col = max(0, col-1) # Left
        if action == 3: col = min(2, col+1) # Right

        new_state = row * 3 + col

        reward = -1 # Small penalty for time wasting
        done = False

        if new_state == self.goal:
            reward = 10
            done = True
        elif new_state in self.pits:
            reward = -10
            done = True

        self.state = new_state
        return new_state, reward, done

def train_q_learning():
    env = GridWorld()
    q_table = np.zeros((env.n_states, env.n_actions))
    alpha = 0.1 # Learning Rate
    gamma = 0.9 # Discount Factor (Future value)
    epsilon = 0.1 # Exploration rate

    print("Training Agent...")
    for episode in range(500):
        env.state = 0
        done = False

        while not done:
            state = env.state

            # Epsilon-Greedy Policy
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, 4) # Explore
            else:
                action = np.argmax(q_table[state]) # Exploit

            next_state, reward, done = env.step(action)

            # The Q-Learning Update Formula (Bellman Equation)
            # Q_new = Q_old + lr * (Reward + discount * max(Q_next) - Q_old)
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state, action] = new_value

    print("\nLearned Q-Table (Value of taking Action X at State Y):")
    # Action 0=Up, 1=Down, 2=Left, 3=Right
    print(q_table)
    
    print("\nOptimal Path inferred from Q-Table:")
    curr = 0
    print(f"Start: {curr}", end="")
    while curr != 8 and curr != 4:
        action = np.argmax(q_table[curr])
        env.state = curr
        curr, _, _ = env.step(action)
        print(f" -> {curr}", end="")
    print(" (Goal!)")

if __name__ == "__main__":
    train_q_learning()
