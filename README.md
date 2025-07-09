## Reinforcement Learning to Develop Policies for Fair and Productive Employment: A Case Study on Wage Theft within the Day-Laborer Community

This repository contains the code for a reinforcement learning simulation by Matt Kammer-Kerwick and Evan Aldrich, designed for PLOS Complex Systems Context. The simulation supports both **single-agent** and **multi-agent** (two-player) logic, allowing for flexible experimentation with different agent interactions.

---

## Features

- **Multi-Agent and Single-Agent Modes:**  
  Switch between a single-agent Markov Decision Process (MDP) and a multi-agent game-theoretic setting by changing a single variable.
- **Customizable Environment:**  
  The [`CustomEnv`](c:/Users/Evan/Documents/GitHub/RL-ProductiveEmployment/S1_file.py) class encapsulates the environment logic, including state transitions, rewards, and stochastic events.
- **Q-Learning Implementation:**  
  Supports standard Q-learning for single-agent and alternating Q-learning for multi-agent settings.
- **Sensitivity Analysis:**  
  Built-in routines for analyzing the effects of learning parameters (`alpha`, `gamma`, `epsilon`) and environment parameters (`pTheft`, `propDegreeTheft`).
- **Visualization:**  
  Plots Q-value distributions and convergence diagnostics for both agents.

---

## How to Use

### 1. Switching Between Single-Agent and Multi-Agent Logic

The simulation mode is controlled by the `agent_type` variable at the top of [`S1_file.py`](c:/Users/Evan/Documents/GitHub/RL-ProductiveEmployment/S1_file.py):

````python
SINGLE_AGENT = 0
MULTI_AGENT = 1

agent_type = MULTI_AGENT  # Set to SINGLE_AGENT or MULTI_AGENT
````

- **`SINGLE_AGENT`**: The environment models a single worker making decisions in a stochastic labor market.
- **`MULTI_AGENT`**: The environment models both an employer and a laborer, each with their own actions and rewards.

### 2. Key Differences Between Modes

| Aspect                | Single-Agent (`SINGLE_AGENT`) | Multi-Agent (`MULTI_AGENT`) |
|-----------------------|-------------------------------|----------------------------|
| **States**            | `idle`, `workingFair`, `workingTheft` | `idle`, `offer_made`, `working`, `stolen` |
| **Actions**           | Varies by state, e.g., `accept`, `decline`, `report` | Employer: `offer`, `no_offer`, `steal`, etc.<br>Laborer: `accept`, `reject`, `report`, etc. |
| **Step Function**     | [`CustomEnv.step`](c:/Users/Evan/Documents/GitHub/RL-ProductiveEmployment/S1_file.py) | [`CustomEnv.stepGAME`](c:/Users/Evan/Documents/GitHub/RL-ProductiveEmployment/S1_file.py) |
| **Reward Structure**  | Single reward per step        | Separate rewards for employer and laborer |
| **Q-Learning Update** | [`reinforcement_learning`](c:/Users/Evan/Documents/GitHub/RL-ProductiveEmployment/S1_file.py) | [`reinforcement_learning_combined`](c:/Users/Evan/Documents/GitHub/RL-ProductiveEmployment/S1_file.py) (alternating updates for both agents) |
| **Experience Sampling** | [`sample_experience`](c:/Users/Evan/Documents/GitHub/RL-ProductiveEmployment/S1_file.py) | [`sample_experience2`](c:/Users/Evan/Documents/GitHub/RL-ProductiveEmployment/S1_file.py) (alternates between agents) |

### 3. Running the Simulation

1. **Set the agent type** at the top of the file.
2. **Run the script**:  
   The main loop will execute Q-learning, print convergence diagnostics, and plot results.
3. **Adjust parameters** as needed for sensitivity analysis or to explore different scenarios.

---

## File Overview

- [`S1_file.py`](c:/Users/Evan/Documents/GitHub/RL-ProductiveEmployment/S1_file.py): Main simulation code, including environment, Q-learning, and plotting routines.

---

## Citing

If you use this code in your research, please cite the associated paper.

---

## Contact

For questions or contributions, please open an issue or contact the repository maintainer.

---
