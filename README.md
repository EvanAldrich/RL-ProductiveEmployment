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

Here is your LaTeX code fully converted into **Markdown** with math formatting preserved using `$$` for display math and `$` for inline math:

---

# Python Workflow.

## Helper Functions

### Daily Pay Function

The daily pay \$P\_{\text{daily}}\$ is drawn from a uniform distribution between a minimum and maximum value:

$$
P_{\text{daily}} \sim U(\text{min pay}, \text{max pay})
$$

Where:

* \$\text{min pay}\$ is the minimum daily pay.
* \$\text{max pay}\$ is the maximum daily pay.

### Degree of Theft Function

The degree of theft \$\delta_T\$ is a random fraction of the potential theft amount, determined by a probability parameter:

$$
\delta_T = R \cdot P_{\text{theft degree}}
$$

Where:

* \$R\$ is a random number such that \$R \in \[0, 1)\$.
* \$P\_{\text{theft degree}}\$ is the maximum proportional degree of theft (e.g., `propDegreeTheft`).

## Custom Environment Dynamics

The environment transitions from a state \$s\$ to a next state \$s'\$ after an action \$a\$ is taken, yielding a reward \$r\$. In the multi-agent setting, two rewards are generated: \$r\_1\$ for the Laborer and \$r\_2\$ for the Employer.

### Multi-Agent Environment Rewards (`stepGAME`)

Let:

* \$C\$ be the cost of being idle (`self.cost`)
* \$P\_{\text{min}}\$, \$P\_{\text{max}}\$ be the min/max daily pay
* \$M\$ be the employer's markup (`self.Markup`)
* \$\delta\_T\$ be the degree of theft
* \$P\_{\text{report success}}\$ be the probability of successful report
* \$E\_{PS}\$ be expected pain and suffering
* \$E\_{DI}\$ be expected days idle

#### Action: `"no_offer"` (from `"idle"`)

$$
r_1 = -C \\
r_2 = -C
$$

#### Action: `"reject"` (from `"offer_made"`)

$$
r_1 = -C \cdot E_{DI} \\
r_2 = -C \cdot E_{DI}
$$

#### Action: `"steal"` (from `"working"`)

Let \$P\_{\text{daily}} \sim U(P\_{\text{min}}, P\_{\text{max}})\$.

$$
r_1 = P_{\text{daily}} \cdot (1 - \delta_T) - C \\
r_2 = (P_{\text{daily}} \cdot M) - (P_{\text{daily}} \cdot (1 - \delta_T))
$$

#### Action: `"no_steal"` (from `"working"`)

Let \$P\_{\text{daily}} \sim U(P\_{\text{min}}, P\_{\text{max}})\$.

$$
r_1 = P_{\text{daily}} - C \\
r_2 = (P_{\text{daily}} \cdot M) - P_{\text{daily}}
$$

#### Action: `"report"` (from `"stolen"`)

Let \$P\_{\text{daily}} \sim U(P\_{\text{min}}, P\_{\text{max}})\$.

If \$\text{rand}() \le P\_{\text{report success}}\$:

$$
r_1 = (P_{\text{daily}} \cdot \delta_T) + E_{PS} \\
r_2 = -(P_{\text{daily}} \cdot \delta_T) - E_{PS}
$$

Else:

$$
r_1 = 0 \\
r_2 = 0
$$

#### Other Actions leading to Next State

For `"offer"` (from `"idle"`) and `"accept"` (from `"offer_made"`), the immediate rewards are:

$$
r_1 = 0 \\
r_2 = 0
$$

## Single-Agent Environment Rewards (`step`)

Let \$P\_{\text{theft}}\$ be the probability of theft.

### Action: `"noAction"` (from `"idle"`)

$$
r = -C
$$

### Action: `"decline"` (from `"idle"`)

$$
r = -C \cdot E_{DI}
$$

### Action: `"accept"` (from `"idle"`)

Let \$P\_{\text{daily}} \sim U(P\_{\text{min}}, P\_{\text{max}})\$.

If \$\text{rand}() \le P\_{\text{theft}}\$:

* Next State = `"workingTheft"`
* $r = P_{\text{daily}} \cdot (1 - \delta_T) - C$

Else:

* Next State = `"workingFair"`
* $r = P_{\text{daily}} - C$

### Action: `"report"` (from `"workingTheft"`)

Let \$P\_{\text{daily}} \sim U(P\_{\text{min}}, P\_{\text{max}})\$.

If \$\text{rand}() \le P\_{\text{report success}}\$:

$$
r = (P_{\text{daily}} \cdot \delta_T) + E_{PS}
$$

Else:

$$
r = 0
$$

### Other Actions leading to Next State

For `"noAction"` from `"workingFair"` and `"workingTheft"`, the immediate reward is:

$$
r = 0
$$

## Reinforcement Learning Algorithms

Let \$Q(s, a)\$ be the Q-value for taking action \$a\$ in state \$s\$, \$r\$ the immediate reward, \$s'\$ the next state, \$\alpha\$ the learning rate, and \$\gamma\$ the discount factor.

### Single-Agent Q-Learning (`reinforcement_learning`)

The Q-value update rule is:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]
$$

### Multi-Agent Q-Learning (`reinforcement_learning_combined`)

Two Q-functions are maintained: \$Q\_1(s, a)\$ for the Laborer and \$Q\_2(s, a)\$ for the Employer.

Let \$r\_1\$, \$r\_2\$ be rewards for Laborer and Employer.

#### Case 1: Employer's policy determines next action

(When `use_Q2 = True`)

$$
Q_2(s, a) \leftarrow Q_2(s, a) + \alpha \left[r_2 + \gamma \max_{a'} Q_2(s', a') - Q_2(s, a)\right] \\
Q_1(s, a) \leftarrow Q_1(s, a) + \alpha \left[r_1 + \gamma Q_1(s', \arg\max_{a''} Q_2(s', a'')) - Q_1(s, a)\right]
$$

#### Case 2: Laborer's policy determines next action

(When `use_Q2 = False`)

$$
Q_1(s, a) \leftarrow Q_1(s, a) + \alpha \left[r_1 + \gamma \max_{a'} Q_1(s', a') - Q_1(s, a)\right] \\
Q_2(s, a) \leftarrow Q_2(s, a) + \alpha \left[r_2 + \gamma Q_2(s', \arg\max_{a''} Q_1(s', a'')) - Q_2(s, a)\right]
$$

If no feasible actions from \$s'\$, use:

$$
\gamma \max_{a'} Q(s', a') = 0
$$

### Epsilon-Greedy Policy

Action \$a\$ from state \$s\$ is selected via:

$$
a =
\begin{cases}
\text{random action from } \mathcal{A}(s) & \text{if } R < \epsilon \\
\arg\max_{a' \in \mathcal{A}(s)} Q(s, a') & \text{if } R \ge \epsilon
\end{cases}
$$

Where:

* \$\mathcal{A}(s)\$ is the set of feasible actions in \$s\$
* \$R \sim U(0,1)\$
* \$\epsilon\$ is the exploration rate

### Learning Rate and Exploration Decay

Decay rules:

$$
\alpha \leftarrow \max(\alpha_{\text{min}}, \alpha \cdot \text{decay rate}) \\
\epsilon \leftarrow \max(\epsilon_{\text{min}}, \epsilon \cdot \text{decay rate})
$$

## Convergence Analysis

### Bootstrap Standard Error

Estimated via resampling:

$$
SE(\bar{Q}) = \sqrt{\frac{1}{N_{\text{boot}} - 1} \sum_{i=1}^{N_{\text{boot}}} (\bar{Q}_i - \bar{\bar{Q}})^2}
$$

Where:

* \$\bar{Q}\$: mean of original Q-values
* \$N\_{\text{boot}}\$: number of bootstrap samples
* \$\bar{Q}\_i\$: mean of \$i\$-th sample
* \$\bar{\bar{Q}}\$: mean of all \$\bar{Q}\_i\$

95% confidence interval:

$$
\bar{Q} \pm 1.96 \cdot SE(\bar{Q})
$$

### Convergence Metrics

Sliding window of size \$W\$:

* **Mean Absolute Change**:

  $$
  \text{MeanChange}_t = \frac{1}{W} \sum_{k=t}^{t+W-1} |Q_k(s, a) - Q_{k-1}(s, a)| < \tau_{\text{mean}}
  $$

* **Variance of Change**:

  $$
  \text{VarChange}_t = \text{Var}(\{|Q_k(s, a) - Q_{k-1}(s, a)|\}_{k=t}^{t+W-1}) < \tau_{\text{var}}
  $$

* **Percentage Stable**:

  $$
  \text{PercStable}_t = \frac{\sum_{k=t}^{t+W-1} \mathbb{I}(|Q_k(s, a) - Q_{k-1}(s, a)| < \tau_{\text{stable}})}{W} > 0.95
  $$

Where \$\mathbb{I}(\cdot)\$ is the indicator function.

---

## Citing

If you use this code in your research, please cite the associated paper.

---

## Contact

For questions or contributions, please open an issue or contact the repository maintainer.

---
