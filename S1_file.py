"""
Reinforcement Learning Simulation for Probabilistic Decision-Making
Adapted and Refactored for Academic Use (PLOS Computational Biology Context)
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import t
# Set seed for random number generator
random.seed(300)

SINGLE_AGENT = 0
MULTI_AGENT = 1

agent_type = MULTI_AGENT # Set to SINGLE_AGENT or MULTI_AGENT to run respective functions

# Helper function: returns a random daily pay between min and max
def dayPay(min, max):
    return (min + random.random() * (max - min))

# Helper function: returns a random theft degree up to prob
def degreeTheft(prob):
    return random.random() * prob

# Action sets for each state in the multi-agent game
actionsByStateGAME = {'idle': ["offer", "no_offer"],
                      'offer_made': ["accept", "reject"],
                      'working': ["steal", "no_steal"],
                      'stolen': ["report", "no_report"]}

# Action sets for each state in the single-agent model
actionsByState = {'idle': ["accept", "decline", "noAction"],
                  'workingFair': ["noAction"],
                  'workingTheft': ["noAction", "report"]}



class CustomEnv:
    """
    Custom environment for the labor market game.
    Handles both single-agent and multi-agent settings.
    """
    def __init__(self):
        self.state = None
        self.event = None
        # States and actions for the multi-agent game
        self.statesGAME = ["idle", "offer_made", "working", "stolen"] 
        self.actionsGAME = ["offer", "no_offer", "accept", "reject", "steal", "no_steal", "report", "no_report"]
        # States and actions for the single-agent model
        self.states = ["idle", "workingFair", "workingTheft"]
        self.actions = ["noAction", "accept", "decline", "report"]
        # Environment parameters (Change these to adjust the game)
        self.cost = 60
        self.min_pay = 80
        self.Markup = 1.2     #value of job = percieved_payout * 1.2 (markup)perceived_payout = perceived_pay * labor_days; perceived_pay = Uniform random distribution based on Job pay max and Job pay min
        self.max_pay = 150
        self.pOfferWork = 0.5
        self.pTheft = 0.3
        self.propDegreeTheft = 0.25   
        self.pReportSuccess = 0.01 #probability of successful report (Policy of intrest)
        self.expectedPainSuffering = 500
        self.ExpectedDaysIdle = self.pOfferWork / (1 - self.pOfferWork)
        self.modelPenalty = -999999
        self.initial_state = "idle"

    def stepGAME(self, action):
        next_state = self.state
        reward, reward2 = 0, 0    #Reward2 = Employer, Reward = Laborer
        done = False

        # State transitions and rewards for each action
        if self.state == "idle" and action == "offer":
            self.event = "JobOffer"
            next_state = "offer_made"
            reward = 0
            reward2 = 0 
        elif self.state == "idle" and action == "no_offer":
            self.event = "Job_not_Offered"
            next_state = "idle" 
            reward = -self.cost
            reward2 = -self.cost
        elif self.state == "offer_made" and action == "reject":
            self.event = "jobOfferDecline"
            next_state = "idle"
            reward = -self.cost * self.ExpectedDaysIdle
            reward2 = -self.cost * self.ExpectedDaysIdle
        elif self.state == "offer_made" and action == "accept":
            self.event = "jobBegin"
            next_state = "working"
            reward = 0 
            reward2 = 0 
        elif self.state == "working" and action == "steal":
            self.event = "jobOfferTheft"
            next_state = "stolen"
            pay = dayPay(self.min_pay, self.max_pay) 
            payout = pay*(1 - degreeTheft(self.propDegreeTheft))
            reward = payout - self.cost
            reward2 = pay*self.Markup - payout 
        elif self.state == "working" and action == "no_steal":
            self.event = "jobOfferFair"
            next_state = "idle"
            pay = dayPay(self.min_pay, self.max_pay)
            reward = pay - self.cost
            reward2 = pay*self.Markup - pay
            done = True
        elif self.state == "stolen" and action == "no_report":
            self.event = "jobEnds"
            next_state = "idle"
            reward = 0
            reward2 = 0
            done = True
        elif self.state == "stolen" and action == "report":
            # Reporting theft: rare chance of success
            if random.random() <= self.pReportSuccess:
                self.event = "jobEnds"
                next_state = "idle"
                reward = dayPay(self.min_pay, self.max_pay) * (
                        degreeTheft(self.propDegreeTheft)) + self.expectedPainSuffering
                reward2 = - dayPay(self.min_pay, self.max_pay) * (degreeTheft(self.propDegreeTheft)) - self.expectedPainSuffering
                done = True
            else:
                self.event = "jobEnds"
                next_state = "idle"
                reward = 0
                reward2 = 0
                done = True
        else:
            # Invalid transition
            reward = self.modelPenalty
            reward2 = self.modelPenalty

        self.state = next_state
        info = {}
        return next_state, reward, reward2, done, info
    
    def step(self, action):
        """
        Step function for the single-agent model.
        Returns next_state, reward, done, info.
        """
        next_state = self.state
        reward = 0
        done = False
        # State transitions and rewards for each action
        if self.state == "idle" and action == "noAction":
            self.event = "JobOffer"
            next_state = "idle"
            reward = -self.cost
        elif self.state == "idle" and action == "decline":
            self.event = "jobOfferDecline"
            next_state = "idle"
            reward = -self.cost * self.ExpectedDaysIdle
        elif self.state == "idle" and action == "accept":
            # Random chance of theft (probabilistic unlike the game version)
            if random.random() <= self.pTheft:
                self.event = "jobOfferTheft"
                next_state = "workingTheft"
                reward = dayPay(self.min_pay, self.max_pay) * (
                        1 - degreeTheft(self.propDegreeTheft)) - self.cost
            else:
                self.event = "jobOfferFair"
                next_state = "workingFair"
                reward = dayPay(self.min_pay, self.max_pay) - self.cost
        elif self.state == "workingFair" and action == "noAction":
            self.event = "jobEnds"
            next_state = "idle"
            reward = 0
            done = True
        elif self.state == "workingTheft" and action == "noAction":
            self.event = "jobEnds"
            next_state = "idle"
            reward = 0
            done = True
        elif self.state == "workingTheft" and action == "report":
            # Reporting theft: rare chance of success
            if random.random() <= self.pReportSuccess:
                self.event = "jobEnds_Successfully"
                next_state = "idle"
                reward = dayPay(self.min_pay, self.max_pay) * (
                        degreeTheft(self.propDegreeTheft)) + self.expectedPainSuffering
                done = True
            else:
                self.event = "jobEnds"
                next_state = "idle"
                reward = 0
                done = True
        else:
            # Invalid transition
            reward = self.modelPenalty

        self.state = next_state
        info = {}
        return next_state, reward, done, info

    def reset(self):
        """
        Reset environment to initial state.
        """
        self.state = self.initial_state
        return self.state

    def render(self, mode='human'):
        pass

def sample_experience_raw(env, num_samples):
    """
    Collects raw experience samples from the environment.
    Used to learn the feasible state-action space.
    """
    data = []
    if agent_type == SINGLE_AGENT:
        actionStateDict = actionsByState.copy()
    else:
        actionStateDict = actionsByStateGAME.copy()
    for _ in range(num_samples):
        state = env.reset()
        done = False
        while not done:
            actionSet4ThisState = actionStateDict[state]
            action = random.choice(actionSet4ThisState)
            if agent_type == SINGLE_AGENT:
                next_state, reward, done, _ = env.step(action)
                if done == True:
                    finish = done
                else:
                    finish = ""
                data.append({"State": state, "Action": action, "Reward": reward, "NextState": next_state})
            else:
                next_state, reward, reward2, done, _ = env.stepGAME(action)
                if done == True:
                    finish = done
                else:
                    finish = ""
                data.append({"State": state, "Action": action, "Reward": reward, "Reward2": reward2, "NextState": next_state})
            state = next_state
    return data

def learn_state_space(df):
    """
    Learns the feasible actions for each state from experience data.
    Returns a dictionary mapping state -> list of actions.
    """
    state_action_dict = {}

    for row in df:
        state = row['State']
        action = row['Action']

        if state not in state_action_dict:
            state_action_dict[state] = []  # Create a new list if state encountered first time

        if action not in state_action_dict[state]:
            state_action_dict[state].append(action)


    return state_action_dict


def sample_experience(env, num_samples, epsilon, feasible_actions, Q_model=None):
    """
    Samples experience using epsilon-greedy policy.
    Returns a list of experience dictionaries.
    """
    model = {} if Q_model is None else Q_model.copy()
    data = []
    for _ in range(num_samples):
        state = env.reset()
        done = False
        while not done:
            actionSet4ThisState = feasible_actions.get(state, [])
            # Epsilon-greedy action selection
            if random.random() <= epsilon:
                action = random.choice(actionSet4ThisState)
            else:
                action = max(actionSet4ThisState, key=lambda a: model.get((state, a), 0))
            
            if agent_type == SINGLE_AGENT:
                next_state, reward, done, _ = env.step(action)
                if done == True:
                    finish = done
                else:
                    finish = ""
                data.append({"State": state, "Action": action, "Reward": reward, "NextState": next_state})
            else:
                next_state, reward, reward2, done, _ = env.stepGAME(action)
                if done == True:
                    finish = done
                else:
                    finish =""
                data.append({"State": state, "Action": action, "Reward": reward, "Reward2": reward2, "NextState": next_state})
            state = next_state
    return data

## new sample to get them in the same universe
def sample_experience2(env, num_samples, epsilon, feasible_actions, Q_model1=None, Q_model2=None):
    """
    Samples experience for both agents in the multi-agent setting.
    Alternates between employer and laborer Q-models.
    Returns two lists of experience.
    """
    model1 = {} if Q_model1 is None else Q_model1.copy()
    model2 = {} if Q_model2 is None else Q_model2.copy()
    data1 = []
    data2= []

    for _ in range(num_samples):
        state = env.reset()
        done = False
        use_q2 = 1 # employer decesion goes first
        while not done:
            actionSet4ThisState = feasible_actions.get(state, [])
            if random.random() <= epsilon:
                action = random.choice(actionSet4ThisState)
            else:
                if use_q2 == 1:
                    action = max(actionSet4ThisState, key=lambda a: model2.get((state, a), 0))
                else:
                    action = max(actionSet4ThisState, key=lambda a: model1.get((state, a), 0))
            
            next_state, reward, reward2, done, _ = env.stepGAME(action)
            if done == True:
                finish = done
            else:
                finish =""
            data1.append({"State": state, "Action": action, "Reward": reward, "Reward2": reward2,  "NextState": next_state})
            data2.append({"State": state, "Action": action, "Reward": reward, "Reward2": reward2,  "NextState": next_state})
            state = next_state
            use_q2 = not use_q2
    return data1, data2


def reinforcement_learning_combined(data, alpha, gamma, feasible_actions, Q1_model=None, Q2_model=None):
    """
    Q-learning update for multi-agent setting.
    Alternates between updating Q1 (laborer) and Q2 (employer).
    """
    Q1 = {} if Q1_model is None else Q1_model.copy()
    Q2 = {} if Q2_model is None else Q2_model.copy()
    use_Q2 = False  # Initially set to use Q1

    for experience in data:
        state = experience["State"]
        action = experience["Action"]
        reward = experience["Reward"]
        reward2 = experience["Reward2"]
        next_state = experience["NextState"]
        next_actions = feasible_actions.get(next_state, [])
        
        if next_actions:
            if use_Q2:
                # Choose the action that maximizes reward2 (use Q2)
                best_next_action_Q2 = max(next_actions, key=lambda a: Q2.get((next_state, a), 0))
                Q2[(state, action)] = Q2.get((state, action), 0) + alpha * (
                    reward2 + gamma * Q2.get((next_state, best_next_action_Q2), 0) - Q2.get((state, action), 0))
                Q1[(state, action)] = Q1.get((state, action), 0) + alpha * (
                    reward + gamma * Q1.get((next_state, best_next_action_Q2), 0) - Q1.get((state, action), 0))
            else:
                # Choose the action that maximizes reward (use Q1)
                best_next_action_Q1 = max(next_actions, key=lambda a: Q1.get((next_state, a), 0))
                Q1[(state, action)] = Q1.get((state, action), 0) + alpha * (
                    reward + gamma * Q1.get((next_state, best_next_action_Q1), 0) - Q1.get((state, action), 0))
                Q2[(state, action)] = Q2.get((state, action), 0) + alpha * (
                    reward2 + gamma * Q2.get((next_state, best_next_action_Q1), 0) - Q2.get((state, action), 0))
        else:
            
            # No feasible next actions, so update only with current reward2
            Q2[(state, action)] = Q2.get((state, action), 0) + alpha * (reward2 - Q2.get((state, action), 0))
            Q1[(state, action)] = Q1.get((state, action), 0) + alpha * (reward - Q1.get((state, action), 0))
        
        use_Q2 = not use_Q2  # Alternate between Q1 and Q2 for each experience

    return Q1, Q2



def reinforcement_learning(data, alpha, gamma, feasible_actions, Q_model=None):
    """
    Standard Q-learning update for single-agent setting.
    """
    Q = {} if Q_model is None else Q_model.copy()
    for experience in data:
        state = experience["State"]
        action = experience["Action"]
        reward = experience["Reward"]
        next_state = experience["NextState"]
        next_actions = feasible_actions.get(next_state, [])
        if next_actions:
            max_q_next = max(Q.get((next_state, a), 0) for a in next_actions)
        else:
            max_q_next = 0  # Default value if there are no next actions
        # Q-learning update rule
        Q[(state, action)] = Q.get((state, action), 0) + alpha * (
                reward + gamma * max_q_next - Q.get((state, action), 0))
    return Q

def print_q_values(Q):
    state_action_pairs = []
    q_values = []

    for (state, action), q_value in Q.items():
        state_action_pairs.append((state, action))
        q_values.append(q_value)

    q_table = pd.DataFrame({'Q-Value': q_values},
                           index=pd.MultiIndex.from_tuples(state_action_pairs, names=['State', 'Action']))

    print(q_table)


def track_q_values(Q_history, Q, iteration):
    """
    Tracks Q-values for plotting and convergence analysis.
    """
    for (state, action), value in Q.items():
        if (state, action) not in Q_history:
            Q_history[(state, action)] = []
        Q_history[(state, action)].append((iteration, value))

# Convergence Flow
raw_data = sample_experience_raw(CustomEnv(), num_samples=10)
feasible_actions = learn_state_space(raw_data)

# Set initial parameters
initial_alpha = 0.1
alpha = initial_alpha
min_alpha = 0.01
gamma = 0.9
initial_epsilon = 0.5
min_epsilon = 0.01
epsilon_decay = 0.99
epsilon = initial_epsilon
max_samples = 100000
sample_step = 100
current_samples = sample_step

# Initialize a list to store rewards over episodes
rewards_history = []
rewards_history_employer = []
sample_df = pd.DataFrame()
sample_df_employer = pd.DataFrame()

# Initialize Q-values and history tracking
Q_model1 = None
if agent_type == MULTI_AGENT:
    Q_model1, Q_model2 = reinforcement_learning_combined(raw_data, alpha, gamma, feasible_actions)
else:
    Q_model1 = reinforcement_learning(raw_data,alpha,gamma,feasible_actions)
Q_history1 = {}
track_q_values(Q_history1, Q_model1, 0)

if agent_type == MULTI_AGENT:
    Q_history2 = {}
    track_q_values(Q_history2, Q_model2, 0)

# Compute initial average reward
rewards = sum(exp['Reward'] for exp in raw_data) / sample_step
rewards_history.append(rewards)
rewards_history_employer.append(rewards)
for exp in raw_data:
    exp['Iteration'] = 0

sample_df = sample_df._append(raw_data, ignore_index=True)
sample_df_employer = sample_df_employer._append(raw_data, ignore_index=True)

print("Before Laborer")
print_q_values(Q_model1)
if agent_type == MULTI_AGENT:
    print("Before Employer")
    print_q_values(Q_model2)

iteration = 1

# Main learning loop: sample experience and update Q-values until convergence
while True:
    # Sample experience
    data1 = sample_experience(CustomEnv(), sample_step, epsilon, feasible_actions, Q_model1)
    for exp in data1:
        exp['Iteration'] = iteration
    sample_df = sample_df._append(data1, ignore_index=True)
    if agent_type == MULTI_AGENT:
        data2 = sample_experience(CustomEnv(), sample_step, epsilon, feasible_actions, Q_model2)
        for exp in data2:
            exp['Iteration'] = iteration
        sample_df_employer = sample_df_employer._append(data2, ignore_index=True)
    
    # Perform reinforcement learning update
    if agent_type == MULTI_AGENT:
        Q_model_updated1, Q_model_updated2 = reinforcement_learning_combined(data2, alpha, gamma, feasible_actions, Q_model1, Q_model2)                   #changes to data2
    else:
        Q_model_updated1 = reinforcement_learning(data1,alpha,gamma,feasible_actions,Q_model1)
    track_q_values(Q_history1, Q_model_updated1, iteration)
    if agent_type == MULTI_AGENT:
        track_q_values(Q_history2, Q_model_updated2, iteration)

    # Update rewards history
    rewards = sum(exp['Reward'] for exp in data1) / len(data1)
    rewards_history.append(rewards)

    if agent_type == MULTI_AGENT:
        rewards = sum(exp['Reward'] for exp in data2) / len(data2)
        rewards_history_employer.append(rewards)

    
    # Update epsilon and alpha (decay)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    alpha = max(min_alpha, alpha * epsilon_decay)

    # Update Q-values and increase sample size
    Q_model1 = Q_model_updated1
    if agent_type == MULTI_AGENT:
        Q_model2 = Q_model_updated2
    

    current_samples += sample_step
    iteration += 1
    if current_samples > max_samples:
        break

print_q_values(Q_model_updated1)
if agent_type == MULTI_AGENT:
    print_q_values(Q_model_updated2)

# Convert rewards_history and iteration to lists for plotting
iterations = list(range(len(rewards_history)))


def bootstrap_std_error(data, num_bootstrap=1000):
    """
    Compute bootstrap standard error for a 1D array.
    """
    n = len(data)
    means = []
    for _ in range(num_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    return np.std(means)


def plot_q_values_distribution(Q_history, label, num_iterations, num_bootstrap=1000):
    """
    Plots Q-value distributions for a specific state and agent label.
    Shows mean and confidence intervals using bootstrapping.
    """
    # Only plot 'stolen' for Laborer and 'working' for Employer
    if agent_type == MULTI_AGENT:
        if label.lower() == 'laborer':
            plot_states = {'stolen'}
        elif label.lower() == 'employer':
            plot_states = {'working'}
        else:
            print("Unknown label. Please use 'Laborer' or 'Employer'.")
            return
    else:
        plot_states = {'workingTheft'}

    # Font sizes
    title_fontsize = 30
    axis_label_fontsize = 25
    ticks_fontsize = 30
    legend_fontsize = 30

    for state in plot_states:
        plt.figure(figsize=(15, 10))
        for (s, action), values in Q_history.items():
            if s == state:
                q_values = [v for i, v in values if i < num_iterations * 50]
                required_size = 50 * num_iterations
                if len(q_values) < required_size:
                    padding_size = required_size - len(q_values)
                    q_values += [q_values[-1]] * padding_size
                elif len(q_values) > required_size:
                    q_values = q_values[:required_size]
                try:
                    q_values = np.array(q_values).reshape(50, num_iterations)
                except ValueError as e:
                    print(f"Error reshaping array: {e}")
                    return
                means = np.mean(q_values, axis=0)
                boot_se = np.array([bootstrap_std_error(q_values[:, i], num_bootstrap) for i in range(q_values.shape[1])])
                lower = means - 1.96 * boot_se
                upper = means + 1.96 * boot_se
                plt.plot(range(num_iterations), means, label=f"Q({state}, {action})")
                plt.fill_between(range(num_iterations), lower, upper, alpha=0.3)
        plt.title(f"Q-values Distribution for {label} in state {state}", fontsize=title_fontsize)
        plt.xlabel('Iterations', fontsize=axis_label_fontsize)
        plt.ylabel('Q-values ($)', fontsize=axis_label_fontsize)
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)
        plt.legend(fontsize=legend_fontsize)
        plt.show()




def check_convergence_per_pair(Q_history, window=10, q_change_threshold=1e-2, stability_threshold=1e-3):
    """
    Checks convergence for each (state, action) pair in Q_history.
    Prints and returns a dict of convergence iterations for each metric per pair.
    """
    convergence_results = {}
    for (state, action), values in Q_history.items():
        if len(values) < window + 1:
            continue  # Not enough data for this pair
        # Extract Q-values per iteration
        iterations, q_vals = zip(*values)
        q_vals = np.array(q_vals)
        # Compute Q-value changes per step
        q_changes = np.abs(np.diff(q_vals))
        # Compute metrics over a sliding window
        mean_conv_iter = None
        var_conv_iter = None
        perc_conv_iter = None
        for i in range(len(q_changes) - window + 1):
            window_changes = q_changes[i:i+window]
            mean_change = np.mean(window_changes)
            var_change = np.var(window_changes)
            perc_stable = np.mean(window_changes < stability_threshold)
            if mean_conv_iter is None and mean_change < q_change_threshold:
                mean_conv_iter = iterations[i+window]
            if var_conv_iter is None and var_change < q_change_threshold:
                var_conv_iter = iterations[i+window]
            if perc_conv_iter is None and perc_stable > 0.95:
                perc_conv_iter = iterations[i+window]
        convergence_results[(state, action)] = {
            "mean_converged_at": mean_conv_iter,
            "var_converged_at": var_conv_iter,
            "perc_converged_at": perc_conv_iter
        }
        print(f"({state}, {action}): "
              f"Mean < {q_change_threshold} at {mean_conv_iter}, "
              f"Var < {q_change_threshold} at {var_conv_iter}, "
              f">95% stable at {perc_conv_iter}")
    return convergence_results

# Main loop with 100 runs of 1000 iterations each (change as needed)
runs = 1000
iterations_per_run = 1000

Q_model1 = None
Q_model2 = None
Q_history1 = {}
Q_history2 = {}
if agent_type == MULTI_AGENT:
    Q_model1, Q_model2 = reinforcement_learning_combined(raw_data, alpha, gamma, feasible_actions)
else:
    Q_model1 = reinforcement_learning(raw_data, alpha, gamma, feasible_actions)
track_q_values(Q_history1, Q_model_updated1, iteration)
if agent_type == MULTI_AGENT:
    track_q_values(Q_history2, Q_model_updated2, iteration)


for run in range(runs):
    iteration = 1


    for i in range(iterations_per_run):
        if agent_type == MULTI_AGENT:
            data1, data2 = sample_experience2(CustomEnv(), sample_step, epsilon, feasible_actions, Q_model1, Q_model2)
        else:
            data1 = sample_experience(CustomEnv(), sample_step, epsilon, feasible_actions, Q_model1)

        if agent_type == MULTI_AGENT:
            Q_model_updated1, Q_model_updated2 = reinforcement_learning_combined(data1, alpha, gamma, feasible_actions, Q_model1, Q_model2)
        else:
            Q_model_updated1 = reinforcement_learning(data1, alpha, gamma, feasible_actions, Q_model1)
        
        track_q_values(Q_history1, Q_model_updated1, iteration + i)
        Q_model1 = Q_model_updated1
        
        if agent_type == MULTI_AGENT:
            track_q_values(Q_history2, Q_model_updated2,iteration + i)
            Q_model2 = Q_model_updated2


    print("\nChecking convergence per state-action pair for Laborer Q-values:")
    check_convergence_per_pair(Q_history1, window=10, q_change_threshold=1e-2, stability_threshold=1e-3)
    if agent_type == MULTI_AGENT:
        print("\nChecking convergence per state-action pair for Employer Q-values:")
        check_convergence_per_pair(Q_history2, window=10, q_change_threshold=1e-2, stability_threshold=1e-3)

    # Decay epsilon and alpha after each run
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    alpha = max(min_alpha, alpha * epsilon_decay)


# Plot Q-value distributions after all runs
plot_q_values_distribution(Q_history1, "Laborer", num_iterations=iterations_per_run)
if agent_type == MULTI_AGENT:
    plot_q_values_distribution(Q_history2, "Employer", num_iterations=iterations_per_run)

def print_final_q_values(Q_history_employer, Q_history_laborer, window=10):
    """
    Prints the average Q-values for the employer in state 'working' and
    for the laborer in state 'stolen' over the last `window` iterations.
    """
    if agent_type == MULTI_AGENT:
        # Employer: state 'working'
        print("Employer Q-values in state 'working':")
        employer_actions = ['steal', 'no_steal']
        for action in employer_actions:
            key = ('working', action)
            if key in Q_history_employer and len(Q_history_employer[key]) >= window:
                q_vals = [q for (it, q) in Q_history_employer[key][-window:]]
                avg_q = np.mean(q_vals)
                print(f"  Action '{action}': mean Q = {avg_q:.4f}")
            else:
                print(f"  Action '{action}': not enough data")

        # Laborer: state 'stolen'
        print("\nLaborer Q-values in state 'stolen':")
        laborer_actions = ['report', 'no_report']
        for action in laborer_actions:
            key = ('stolen', action)
            if key in Q_history_laborer and len(Q_history_laborer[key]) >= window:
                q_vals = [q for (it, q) in Q_history_laborer[key][-window:]]
                avg_q = np.mean(q_vals)
                print(f"  Action '{action}': mean Q = {avg_q:.4f}")
            else:
                print(f"  Action '{action}': not enough data")
    else:
        # Single agent: state 'workingTheft'
        print("Laborer Q-values in state 'workingTheft':")
        laborer_actions = ['noAction', 'report']
        for action in laborer_actions:
            key = ('workingTheft', action)
            if key in Q_history_laborer and len(Q_history_laborer[key]) >= window:
                q_vals = [q for (it, q) in Q_history_laborer[key][-window:]]
                avg_q = np.mean(q_vals)
                print(f"  Action '{action}': mean Q = {avg_q:.4f}")
            else:
                print(f"  Action '{action}': not enough data")

print_final_q_values(Q_history2, Q_history1, window=10)


"""
All code below is for sensitivity analysis of learning and model paramaters and plotting Q-values for specific states.
"""

# --- Sensitivity analysis for alpha: Q-values for specific states ---
alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
results = []
window = 5  # Averaging window size

laborer_state = 'stolen'
laborer_actions = ['report', 'no_report']
single_actions = ['noAction', 'report']  # For single agent case
employer_state = 'working'
employer_actions = ['steal', 'no_steal']

# For each alpha, store Q-value histories for the laborer and employer
laborer_q_histories = []
employer_q_histories = []

for alpha_test in alphas:
    Q_model1, Q_model2 = None, None
    Q_history1, Q_history2 = {}, {}
    # (Re-)initialize Q-models as in your main loop
    if agent_type == MULTI_AGENT:
        Q_model1, Q_model2 = reinforcement_learning_combined(raw_data, alpha_test, gamma, feasible_actions)
    else:
        Q_model1 = reinforcement_learning(raw_data, alpha_test, gamma, feasible_actions)
    track_q_values(Q_history1, Q_model1, 0)
    if agent_type == MULTI_AGENT:
        track_q_values(Q_history2, Q_model2, 0)
    # Run for a fixed number of iterations (e.g., 1000)
    for i in range(1000):
        if agent_type == MULTI_AGENT:
            data1, data2 = sample_experience2(CustomEnv(), sample_step, epsilon, feasible_actions, Q_model1, Q_model2)
            Q_model_updated1, Q_model_updated2 = reinforcement_learning_combined(data1, alpha_test, gamma, feasible_actions, Q_model1, Q_model2)
        else:
            data1 = sample_experience(CustomEnv(), sample_step, epsilon, feasible_actions, Q_model1)
            Q_model_updated1 = reinforcement_learning(data1, alpha_test, gamma, feasible_actions, Q_model1)
        Q_model1 = Q_model_updated1
        if agent_type == MULTI_AGENT:
            Q_model2 = Q_model_updated2
        track_q_values(Q_history1, Q_model1, i+1)
        if agent_type == MULTI_AGENT:
            track_q_values(Q_history2, Q_model2, i+1)
    # Collect Q-value histories for plotting
    laborer_q = {}
    if agent_type == MULTI_AGENT:
        for action in laborer_actions:
            key = (laborer_state, action)
            if key in Q_history1:
                laborer_q[action] = [q for (it, q) in Q_history1[key]]
    else:
        for action in single_actions:
            key = ('workingTheft', action)
            if key in Q_history1:
                laborer_q[action] = [q for (it, q) in Q_history1[key]]
    laborer_q_histories.append(laborer_q)
    if agent_type == MULTI_AGENT:
        employer_q = {}
        for action in employer_actions:
            key = (employer_state, action)
            if key in Q_history2:
                employer_q[action] = [q for (it, q) in Q_history2[key]]
        employer_q_histories.append(employer_q)

# --- Laborer Q-values in 'stolen' state ---
if agent_type == SINGLE_AGENT:
    laborer_actions = single_actions  # Use single agent actions
fig, axes = plt.subplots(1, len(laborer_actions), figsize=(7 * len(laborer_actions), 5), sharey=True)
if len(laborer_actions) == 1:
    axes = [axes]  # Ensure axes is iterable

for j, action in enumerate(laborer_actions):
    ax = axes[j]
    for i, alpha_test in enumerate(alphas):
        q_vals = laborer_q_histories[i].get(action, [])
        if len(q_vals) >= window:
            q_avg = np.mean(np.array(q_vals[:len(q_vals)//window*window]).reshape(-1, window), axis=1)
            x_avg = np.arange(len(q_avg)) * window
            ax.plot(x_avg, q_avg, label=f"alpha={alpha_test}")
    ax.set_title(f"Laborer Q-value: action '{action}' in state 'stolen'")
    ax.set_xlabel('Iteration')
    if j == 0:
        ax.set_ylabel("Q-value")
    ax.legend()
plt.suptitle("Sensitivity: Laborer Q-values in 'stolen' state (per action)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Employer Q-values in 'working' state ---
if agent_type == MULTI_AGENT:
    fig, axes = plt.subplots(1, len(employer_actions), figsize=(7 * len(employer_actions), 5), sharey=True)
    if len(employer_actions) == 1:
        axes = [axes]
    for j, action in enumerate(employer_actions):
        ax = axes[j]
        for i, alpha_test in enumerate(alphas):
            q_vals = employer_q_histories[i].get(action, [])
            if len(q_vals) >= window:
                q_avg = np.mean(np.array(q_vals[:len(q_vals)//window*window]).reshape(-1, window), axis=1)
                x_avg = np.arange(len(q_avg)) * window
                ax.plot(x_avg, q_avg, label=f"alpha={alpha_test}")
        ax.set_title(f"Employer Q-value: action '{action}' in state 'working'")
        ax.set_xlabel('Iteration')
        if j == 0:
            ax.set_ylabel("Q-value")
        ax.legend()
    plt.suptitle("Sensitivity: Employer Q-values in 'working' state (per action)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()




# --- Sensitivity analysis for gamma: Q-values for specific states ---
gammas = [0.1, 0.5, 0.75, 0.9, 0.95]
results = []
window = 5  # Averaging window size

# For each gamma, store Q-value histories for the laborer and employer
laborer_q_histories = []
employer_q_histories = []

for gamma_test in gammas:
    Q_model1, Q_model2 = None, None
    Q_history1, Q_history2 = {}, {}
    # (Re-)initialize Q-models as in your main loop
    if agent_type == MULTI_AGENT:
        Q_model1, Q_model2 = reinforcement_learning_combined(raw_data, alpha, gamma_test, feasible_actions)
    else:
        Q_model1 = reinforcement_learning(raw_data, alpha, gamma_test, feasible_actions)
    track_q_values(Q_history1, Q_model1, 0)
    if agent_type == MULTI_AGENT:
        track_q_values(Q_history2, Q_model2, 0)
    # Run for a fixed number of iterations (e.g., 10000)
    for i in range(10000):
        if agent_type == MULTI_AGENT:
            data1, data2 = sample_experience2(CustomEnv(), sample_step, epsilon, feasible_actions, Q_model1, Q_model2)
            Q_model_updated1, Q_model_updated2 = reinforcement_learning_combined(data1, alpha, gamma_test, feasible_actions, Q_model1, Q_model2)
        else:
            data1 = sample_experience(CustomEnv(), sample_step, epsilon, feasible_actions, Q_model1)
            Q_model_updated1 = reinforcement_learning(data1, alpha, gamma_test, feasible_actions, Q_model1)
        Q_model1 = Q_model_updated1
        if agent_type == MULTI_AGENT:
            Q_model2 = Q_model_updated2
        track_q_values(Q_history1, Q_model1, i+1)
        if agent_type == MULTI_AGENT:
            track_q_values(Q_history2, Q_model2, i+1)
    # Collect Q-value histories for plotting
    laborer_q = {}
    if agent_type == MULTI_AGENT:
        for action in laborer_actions:
            key = (laborer_state, action)
            if key in Q_history1:
                laborer_q[action] = [q for (it, q) in Q_history1[key]]
    else:
        for action in laborer_actions:
            key = ('workingTheft', action)
            if key in Q_history1:
                laborer_q[action] = [q for (it, q) in Q_history1[key]]
    laborer_q_histories.append(laborer_q)
    if agent_type == MULTI_AGENT:
        employer_q = {}
        for action in employer_actions:
            key = (employer_state, action)
            if key in Q_history2:
                employer_q[action] = [q for (it, q) in Q_history2[key]]
        employer_q_histories.append(employer_q)

# --- Laborer Q-values in 'stolen' state ---
fig, axes = plt.subplots(1, len(laborer_actions), figsize=(7 * len(laborer_actions), 5), sharey=True)
if len(laborer_actions) == 1:
    axes = [axes]  # Ensure axes is iterable

for j, action in enumerate(laborer_actions):
    ax = axes[j]
    for i, gamma_test in enumerate(gammas):
        q_vals = laborer_q_histories[i].get(action, [])
        if len(q_vals) >= window:
            q_avg = np.mean(np.array(q_vals[:len(q_vals)//window*window]).reshape(-1, window), axis=1)
            x_avg = np.arange(len(q_avg)) * window
            ax.plot(x_avg, q_avg, label=f"gamma={gamma_test}")
    ax.set_title(f"Laborer Q-value: action '{action}' in state 'stolen'")
    ax.set_xlabel('Iteration')
    if j == 0:
        ax.set_ylabel("Q-value")
    ax.legend()
plt.suptitle("Sensitivity: Laborer Q-values in 'stolen' state (per action)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Employer Q-values in 'working' state ---
if agent_type == MULTI_AGENT:
    fig, axes = plt.subplots(1, len(employer_actions), figsize=(7 * len(employer_actions), 5), sharey=True)
    if len(employer_actions) == 1:
        axes = [axes]
    for j, action in enumerate(employer_actions):
        ax = axes[j]
        for i, gamma_test in enumerate(gammas):
            q_vals = employer_q_histories[i].get(action, [])
            if len(q_vals) >= window:
                q_avg = np.mean(np.array(q_vals[:len(q_vals)//window*window]).reshape(-1, window), axis=1)
                x_avg = np.arange(len(q_avg)) * window
                ax.plot(x_avg, q_avg, label=f"gamma={gamma_test}")
        ax.set_title(f"Employer Q-value: action '{action}' in state 'working'")
        ax.set_xlabel('Iteration')
        if j == 0:
            ax.set_ylabel("Q-value")
        ax.legend()
    plt.suptitle("Sensitivity: Employer Q-values in 'working' state (per action)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()





# --- Sensitivity analysis for epsilon: Q-values for specific states ---
epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]
window = 5  # Averaging window size

laborer_q_histories = []
employer_q_histories = []

for epsilon_test in epsilons:
    Q_model1, Q_model2 = None, None
    Q_history1, Q_history2 = {}, {}
    # (Re-)initialize Q-models as in your main loop
    if agent_type == MULTI_AGENT:
        Q_model1, Q_model2 = reinforcement_learning_combined(raw_data, alpha, gamma, feasible_actions)
    else:
        Q_model1 = reinforcement_learning(raw_data, alpha, gamma, feasible_actions)
    track_q_values(Q_history1, Q_model1, 0)
    if agent_type == MULTI_AGENT:
        track_q_values(Q_history2, Q_model2, 0)
    # Run for a fixed number of iterations (e.g., 10000)
    for i in range(10000):
        if agent_type == MULTI_AGENT:
            data1, data2 = sample_experience2(CustomEnv(), sample_step, epsilon_test, feasible_actions, Q_model1, Q_model2)
            Q_model_updated1, Q_model_updated2 = reinforcement_learning_combined(data1, alpha, gamma, feasible_actions, Q_model1, Q_model2)
        else:
            data1 = sample_experience(CustomEnv(), sample_step, epsilon_test, feasible_actions, Q_model1)
            Q_model_updated1 = reinforcement_learning(data1, alpha, gamma, feasible_actions, Q_model1)
        Q_model1 = Q_model_updated1
        if agent_type == MULTI_AGENT:
            Q_model2 = Q_model_updated2
        track_q_values(Q_history1, Q_model1, i+1)
        if agent_type == MULTI_AGENT:
            track_q_values(Q_history2, Q_model2, i+1)
    # Collect Q-value histories for plotting
    laborer_q = {}
    if agent_type == MULTI_AGENT:
        for action in laborer_actions:
            key = (laborer_state, action)
            if key in Q_history1:
                laborer_q[action] = [q for (it, q) in Q_history1[key]]
    else:
        for action in laborer_actions:
            key = ('workingTheft', action)
            if key in Q_history1:
                laborer_q[action] = [q for (it, q) in Q_history1[key]]
    laborer_q_histories.append(laborer_q)
    if agent_type == MULTI_AGENT:
        employer_q = {}
        for action in employer_actions:
            key = (employer_state, action)
            if key in Q_history2:
                employer_q[action] = [q for (it, q) in Q_history2[key]]
        employer_q_histories.append(employer_q)

# --- Laborer Q-values in 'stolen' state (epsilon) ---
fig, axes = plt.subplots(1, len(laborer_actions), figsize=(7 * len(laborer_actions), 5), sharey=True)
if len(laborer_actions) == 1:
    axes = [axes]  # Ensure axes is iterable

for j, action in enumerate(laborer_actions):
    ax = axes[j]
    for i, epsilon_test in enumerate(epsilons):
        q_vals = laborer_q_histories[i].get(action, [])
        if len(q_vals) >= window:
            q_avg = np.mean(np.array(q_vals[:len(q_vals)//window*window]).reshape(-1, window), axis=1)
            x_avg = np.arange(len(q_avg)) * window
            ax.plot(x_avg, q_avg, label=f"epsilon={epsilon_test}")
    ax.set_title(f"Laborer Q-value: action '{action}' in state 'stolen'")
    ax.set_xlabel('Iteration')
    if j == 0:
        ax.set_ylabel("Q-value")
    ax.legend()
plt.suptitle("Sensitivity: Laborer Q-values in 'stolen' state (per action, epsilon)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Employer Q-values in 'working' state (epsilon) ---
if agent_type == MULTI_AGENT:
    fig, axes = plt.subplots(1, len(employer_actions), figsize=(7 * len(employer_actions), 5), sharey=True)
    if len(employer_actions) == 1:
        axes = [axes]
    for j, action in enumerate(employer_actions):
        ax = axes[j]
        for i, epsilon_test in enumerate(epsilons):
            q_vals = employer_q_histories[i].get(action, [])
            if len(q_vals) >= window:
                q_avg = np.mean(np.array(q_vals[:len(q_vals)//window*window]).reshape(-1, window), axis=1)
                x_avg = np.arange(len(q_avg)) * window
                ax.plot(x_avg, q_avg, label=f"epsilon={epsilon_test}")
        ax.set_title(f"Employer Q-value: action '{action}' in state 'working'")
        ax.set_xlabel('Iteration')
        if j == 0:
            ax.set_ylabel("Q-value")
        ax.legend()
    plt.suptitle("Sensitivity: Employer Q-values in 'working' state (per action, epsilon)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



# --- Sensitivity analysis for pTheft and propDegreeTheft ---

# Define the parameter ranges to test
pTheft_values = [0.05, 0.15, 0.3, 0.5, 0.8]
propDegreeTheft_values = [0.05, 0.15, 0.25, 0.5, 0.8]
window = 5  # Averaging window size

# Sensitivity for pTheft
laborer_q_histories = []
employer_q_histories = []

for pTheft_test in pTheft_values:
    # Create a new environment with the tested pTheft
    env = CustomEnv()
    env.pTheft = pTheft_test
    Q_model1, Q_model2 = None, None
    Q_history1, Q_history2 = {}, {}
    # (Re-)initialize Q-models as in your main loop
    if agent_type == MULTI_AGENT:
        Q_model1, Q_model2 = reinforcement_learning_combined(raw_data, alpha, gamma, feasible_actions)
    else:
        Q_model1 = reinforcement_learning(raw_data, alpha, gamma, feasible_actions)
    track_q_values(Q_history1, Q_model1, 0)
    if agent_type == MULTI_AGENT:
        track_q_values(Q_history2, Q_model2, 0)
    # Run for a fixed number of iterations (e.g., 10000)
    for i in range(10000):
        if agent_type == MULTI_AGENT:
            data1, data2 = sample_experience2(env, sample_step, epsilon, feasible_actions, Q_model1, Q_model2)
            Q_model_updated1, Q_model_updated2 = reinforcement_learning_combined(data1, alpha, gamma, feasible_actions, Q_model1, Q_model2)
        else:
            data1 = sample_experience(env, sample_step, epsilon, feasible_actions, Q_model1)
            Q_model_updated1 = reinforcement_learning(data1, alpha, gamma, feasible_actions, Q_model1)
        Q_model1 = Q_model_updated1
        if agent_type == MULTI_AGENT:
            Q_model2 = Q_model_updated2
        track_q_values(Q_history1, Q_model1, i+1)
        if agent_type == MULTI_AGENT:
            track_q_values(Q_history2, Q_model2, i+1)
    # Collect Q-value histories for plotting
    laborer_q = {}
    if agent_type == MULTI_AGENT:
        for action in laborer_actions:
            key = (laborer_state, action)
            if key in Q_history1:
                laborer_q[action] = [q for (it, q) in Q_history1[key]]
    else:
        for action in laborer_actions:
            key = ('workingTheft', action)
            if key in Q_history1:
                laborer_q[action] = [q for (it, q) in Q_history1[key]]
    laborer_q_histories.append(laborer_q)
    if agent_type == MULTI_AGENT:
        employer_q = {}
        for action in employer_actions:
            key = (employer_state, action)
            if key in Q_history2:
                employer_q[action] = [q for (it, q) in Q_history2[key]]
        employer_q_histories.append(employer_q)

# Plotting for pTheft
fig, axes = plt.subplots(1, len(laborer_actions), figsize=(7 * len(laborer_actions), 5), sharey=True)
if len(laborer_actions) == 1:
    axes = [axes]
for j, action in enumerate(laborer_actions):
    ax = axes[j]
    for i, pTheft_test in enumerate(pTheft_values):
        q_vals = laborer_q_histories[i].get(action, [])
        if len(q_vals) >= window:
            q_avg = np.mean(np.array(q_vals[:len(q_vals)//window*window]).reshape(-1, window), axis=1)
            x_avg = np.arange(len(q_avg)) * window
            ax.plot(x_avg, q_avg, label=f"pTheft={pTheft_test}")
    ax.set_title(f"Laborer Q-value: action '{action}' (pTheft)")
    ax.set_xlabel('Iteration')
    if j == 0:
        ax.set_ylabel("Q-value")
    ax.legend()
plt.suptitle("Sensitivity: Laborer Q-values (pTheft)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Repeat for propDegreeTheft ---

laborer_q_histories = []
employer_q_histories = []

for propDegreeTheft_test in propDegreeTheft_values:
    env = CustomEnv()
    env.propDegreeTheft = propDegreeTheft_test
    Q_model1, Q_model2 = None, None
    Q_history1, Q_history2 = {}, {}
    if agent_type == MULTI_AGENT:
        Q_model1, Q_model2 = reinforcement_learning_combined(raw_data, alpha, gamma, feasible_actions)
    else:
        Q_model1 = reinforcement_learning(raw_data, alpha, gamma, feasible_actions)
    track_q_values(Q_history1, Q_model1, 0)
    if agent_type == MULTI_AGENT:
        track_q_values(Q_history2, Q_model2, 0)
    for i in range(1000):
        if agent_type == MULTI_AGENT:
            data1, data2 = sample_experience2(env, sample_step, epsilon, feasible_actions, Q_model1, Q_model2)
            Q_model_updated1, Q_model_updated2 = reinforcement_learning_combined(data1, alpha, gamma, feasible_actions, Q_model1, Q_model2)
        else:
            data1 = sample_experience(env, sample_step, epsilon, feasible_actions, Q_model1)
            Q_model_updated1 = reinforcement_learning(data1, alpha, gamma, feasible_actions, Q_model1)
        Q_model1 = Q_model_updated1
        if agent_type == MULTI_AGENT:
            Q_model2 = Q_model_updated2
        track_q_values(Q_history1, Q_model1, i+1)
        if agent_type == MULTI_AGENT:
            track_q_values(Q_history2, Q_model2, i+1)
    laborer_q = {}
    if agent_type == MULTI_AGENT:
        for action in laborer_actions:
            key = (laborer_state, action)
            if key in Q_history1:
                laborer_q[action] = [q for (it, q) in Q_history1[key]]
    else:
        for action in laborer_actions:
            key = ('workingTheft', action)
            if key in Q_history1:
                laborer_q[action] = [q for (it, q) in Q_history1[key]]
    laborer_q_histories.append(laborer_q)
    if agent_type == MULTI_AGENT:
        employer_q = {}
        for action in employer_actions:
            key = (employer_state, action)
            if key in Q_history2:
                employer_q[action] = [q for (it, q) in Q_history2[key]]
        employer_q_histories.append(employer_q)

# Plotting for propDegreeTheft
fig, axes = plt.subplots(1, len(laborer_actions), figsize=(7 * len(laborer_actions), 5), sharey=True)
if len(laborer_actions) == 1:
    axes = [axes]
for j, action in enumerate(laborer_actions):
    ax = axes[j]
    for i, propDegreeTheft_test in enumerate(propDegreeTheft_values):
        q_vals = laborer_q_histories[i].get(action, [])
        if len(q_vals) >= window:
            q_avg = np.mean(np.array(q_vals[:len(q_vals)//window*window]).reshape(-1, window), axis=1)
            x_avg = np.arange(len(q_avg)) * window
            ax.plot(x_avg, q_avg, label=f"propDegreeTheft={propDegreeTheft_test}")
    ax.set_title(f"Laborer Q-value: action '{action}' (propDegreeTheft)")
    ax.set_xlabel('Iteration')
    if j == 0:
        ax.set_ylabel("Q-value")
    ax.legend()
plt.suptitle("Sensitivity: Laborer Q-values (propDegreeTheft)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

if agent_type == MULTI_AGENT:
    fig, axes = plt.subplots(1, len(employer_actions), figsize=(7 * len(employer_actions), 5), sharey=True)
    if len(employer_actions) == 1:
        axes = [axes]
    for j, action in enumerate(employer_actions):
        ax = axes[j]
        for i, propDegreeTheft_test in enumerate(propDegreeTheft_values):
            # Each employer_q_histories[i] is a dict: action -> list of Q-values
            q_vals = employer_q_histories[i].get(action, [])
            if len(q_vals) >= window:
                q_avg = np.mean(np.array(q_vals[:len(q_vals)//window*window]).reshape(-1, window), axis=1)
                x_avg = np.arange(len(q_avg)) * window
                ax.plot(x_avg, q_avg, label=f"propDegreeTheft={propDegreeTheft_test}")
        ax.set_title(f"Employer Q-value: action '{action}' (propDegreeTheft)")
        ax.set_xlabel('Iteration')
        if j == 0:
            ax.set_ylabel("Q-value")
        ax.legend()
    plt.suptitle("Sensitivity: Employer Q-values (propDegreeTheft, per action)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()