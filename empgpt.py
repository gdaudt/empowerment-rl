import numpy as np

# Define the grid size
GRID_SIZE = 3

# Define the state space as all coordinates in the grid
states = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]

# Define the actions as changes in coordinates
actions = {
    'U': (0, 1),
    'D': (0, -1),
    'L': (-1, 0),
    'R': (1, 0),
    'UL': (-1, 1),
    'UR': (1, 1),
    'DL': (-1, -1),
    'DR': (1, -1)
}

# Probability of choosing each action (uniform distribution)
action_prob = 1 / len(actions)

# Terminal states are defined as states that go out of bounds
def is_terminal(state):
    x, y = state
    return x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE

# Function to get the resulting state from a given state and action
def get_next_state(state, action):
    x, y = state
    dx, dy = action
    next_state = (x + dx, y + dy)
    return next_state

# Function to compute the n-step transition probability for a state-action pair iteratively
def n_step_prob(state, action, n):
    # Step 0: Start with initial state having probability 1
    prob_distribution = {0: {state: 1}}

    # Iterate from 1 to n to build the probability distribution
    for step in range(1, n + 1):
        prob_distribution[step] = {}
        # Iterate over each state and its probability in the previous step
        for s_prev, p_s_prev in prob_distribution[step - 1].items():
            # If the previous state is terminal, it stays there with probability 1
            if is_terminal(s_prev):
                if s_prev not in prob_distribution[step]:
                    prob_distribution[step][s_prev] = 0
                prob_distribution[step][s_prev] += p_s_prev
                continue
            
            # Iterate over all actions to update probabilities
            for a in actions.values():
                next_state = get_next_state(s_prev, a)
                
                # Skip terminal states in the next step
                if is_terminal(next_state):
                    continue
                
                if next_state not in prob_distribution[step]:
                    prob_distribution[step][next_state] = 0
                prob_distribution[step][next_state] += p_s_prev * action_prob
    
    # Return the probability distribution for the nth step for the given action
    result_distribution = {}
    
    # Start from the given state and apply the specified action
    next_state = get_next_state(state, action)
    
    if is_terminal(next_state):
        result_distribution[next_state] = 1
    else:
        if next_state in prob_distribution[n]:
            result_distribution[next_state] = prob_distribution[n][next_state]
        else:
            result_distribution[next_state] = 0
    print(f"n-step probability distribution for state {state} and action {action} at step {n}: {result_distribution}")
    return result_distribution

# Function to compute the value of C for a given state over n steps
def compute_c(state, n):
    C_value = 0

    # Iterate over each action
    for action_name, action in actions.items():
        # Get the n-step probability distribution for the given state and action
        n_step_distribution = n_step_prob(state, action, n)
        
        # Calculate p(S_{n+1} | a_n) for each resulting state
        for s_n1, p_s_n1_given_a_n in n_step_distribution.items():
            # Skip terminal states
            if is_terminal(s_n1):
                continue
            
            # Compute the denominator sum over all actions
            denominator = 0
            for a_prime in actions.values():
                n_step_distribution_prime = n_step_prob(state, a_prime, n)
                denominator += n_step_distribution_prime.get(s_n1, 0) * action_prob
            
            # Skip if denominator is zero
            if denominator == 0:
                continue
            
            # Calculate the log term
            log_term = np.log2(p_s_n1_given_a_n / denominator)
            
            # Add the weighted value to the sum
            C_value += p_s_n1_given_a_n * action_prob
            print(f"Action: {action_name}, State: {state}, Next State: {s_n1}, Log Term: {log_term}")
            print("Equation variables: p_s_n1_given_a_n {}, action_prob {}, log_term {}".format(p_s_n1_given_a_n, action_prob, log_term))
    C_value = C_value * log_term
    print(f"Current C value: {C_value}")
    return C_value

# Example Usage: Calculate C for state (0, 0) with 1-step and 2-step probabilities
initial_state = (1,1)
n_steps = 1
result = compute_c(initial_state, n_steps)
print(f"The value of C for state {initial_state} over {n_steps} steps is: {result}")