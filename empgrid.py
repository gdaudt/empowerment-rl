#print a 3x3 grid with white background and black lines using pygame
import pygame
import sys
from pygame.locals import *
import numpy as np

# set up pygame
pygame.init()

# set up the window, 
WINDOWWIDTH = 1600  
WINDOWHEIGHT = 1600
windowSurface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT), 0, 32)
pygame.display.set_caption('Grid')

# set up the colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

GRID_SIZE = (4, 4)  # 5x5 grid example
WALL = 1            # Marking walls with a value of 1
EMPTY = 0           # Empty cells are where the agent can move

# Example grid (0: empty, 1: wall)
# 3x3 in the middle and walls around the edges
# grid = np.array([
#     [1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 1],
#     [1, 0, 0, 0, 1],
#     [1, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1]
# ])
# 4 x 4 in the middle and walls around the edges
grid = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1]
])

#11x11 grid with walls around the edges and all free spaces
# grid = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ])

# grid with some walls added
# grid = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
#     [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
#     [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
#     [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
#     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ])

gridsize = 10


# Define the 8 diagonal directions and stay-in-place action
actions = [(-1, -1), (-1, 0), (-1, 1),  # Up-left, Up, Up-right
           (0, -1),  (0, 0),  (0, 1),   # Left, Stay, Right
           (1, -1),  (1, 0),  (1, 1)]   # Down-left, Down, Down-right
len_actions = len(actions) - 1
print(len_actions)
#define the 4 cardinal actions as an alternative
# actions = [(-1, 0),  # Up
#            (0, -1),  # Left
#            (0, 1),   # Right
#            (1, 0)]   # Down

k = len(actions) - 1  # Number of actions
action_prob = 1/k  # Probability of each action (equally likely)
print("Action probability: {}".format(action_prob))

# define the partial obstacles in the grid as a list of non-allowed transitions
# for example, agent cannot move from (2,1) to (2,2)
partial_obstacles = [
    # ((1,1), (1,2)),
    # # 1-1 to 2-1
    # ((1,1), (2,1)),
    # # 1-1 to 2-2
    # ((1,1), (2,2))
    ]

# Function to check if a move is valid (not into a wall or out of bounds)
def is_valid(old_state, new_state, grid):
    a, b = old_state
    x, y = new_state
    # check if the state transition is a partial obstacle
    if (old_state, new_state) in partial_obstacles or (new_state, old_state) in partial_obstacles:
        return False
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] != WALL

#print valid values for all cells in the grid
# for i in range(grid.shape[0]):
#     for j in range(grid.shape[1]):
#         print("Cell ({}, {}) is valid: {}".format(i, j, is_valid((i, j), grid)))

# Function to move the agent
def move(state, action, grid):
    new_state = (state[0] + action[0], state[1] + action[1])
    if is_valid(state, new_state, grid):
        return new_state
    return state  # If the move is invalid, stay in place

# Function to move an agent, but differing between terminal and non-terminal states
# if the new state is valid, return the new state with the variable terminal set to False
# if the new state is invalid (would be a collision) return the new state with the variable terminal set to True
def move_terminal(state, action, grid):
    new_state = (state[0] + action[0], state[1] + action[1])
    if is_valid(state, new_state, grid):
        return new_state, False
    return new_state, True  # If the move is invalid, return terminal

def is_terminal(state):
    #return false if state is not a wall or obstacle
    return grid[state[0], state[1]] == WALL

def reachable_states_verbose(state, grid, n):
    reachable = set([state])  # The agent starts in its initial state
    current_states = set([state])
    
    for step in range(n):
        new_states = set()
        for cur_state in current_states:
            for action in actions:
                new_state = move(cur_state, action, grid)
                new_states.add(new_state)
        reachable.update(new_states)
        current_states = new_states  # Proceed to next step
    
    return reachable

def reachable_states(state, grid, n):
    reachable = set([state])  # The agent starts in its initial state
    current_states = set([state])
    
    for step in range(n):
        new_states = set()
        for cur_state in current_states:
            for action in actions:
                new_state = move(cur_state, action, grid)
                new_states.add(new_state)
        reachable.update(new_states)
        current_states = new_states  # Proceed to next step
    
    return len(reachable)

# no need to store the terminal flag, just add the terminal state to the terminal set and the reachable states to reachable set, and return both sets
def reachable_states_terminal_verbose(state, grid, n):
    reachable = {} # agent starts in initial state, reachable states at step 0
    reachable[state] = 0
    terminal = {} # terminal states
    #current states can still be a set
    current_states = set([state])
    step_count = 1
    for step in range(n):
        new_states = set()
        term_states = set()
        for cur_state in current_states:
            for action in actions:
                new_state, term = move_terminal(cur_state, action, grid)
                if term:
                    term_states.add(new_state)
                else:
                    new_states.add(new_state)
        #add each new state to the terminal set, associating it to the step
        for term_state in term_states:
            if(term_state not in terminal):
                terminal[term_state] = step_count
        #do the same for the reachable states
        for new_state in new_states:
            if(new_state not in reachable):
                reachable[new_state] = step_count
        current_states = new_states  # Proceed to next step
        step_count += 1
    
    return reachable, terminal

# Function to compute deterministic empowerment for n steps
def deterministic_empowerment(state, grid, n):
    #print("Reachable states from {}: {}, reachable size is {}.".format(state, reachable, len(reachable)))
    reach_value = reachable_states(state, grid, n) - 1 # Subtract 1 to exclude the initial state
    empowerment = np.log2(reach_value)  # log of the number of unique reachable states (since it's deterministic, it reduces to the log of the number of unique reachable states)
    #print each variable in the equation in the equation format    
    return empowerment

def deterministic_empowerment_terminal(state, grid, n):
    reachable, terminal = reachable_states_terminal_verbose(state, grid, n)
    
    #print the dictionaries with keys and values
    # print("Reachable states from {}: {} at step {}, reachable size is {}.".format(state, reachable[0], reachable[1], len(reachable)))
    # print("Terminal states from {}: {} at step {}, terminal size is {}.".format(state, terminal[0], terminal[1], len(terminal)))
    print("From state: {}".format(state))
    for key, value in reachable.items():
        print("Reachable state: {} at step {}".format(key, value))
    print("reachable size is {}.".format(len(reachable)))
    for key, value in terminal.items():
        print("Terminal state: {} at step {}".format(key, value))
    print("terminal size is {}.".format(len(terminal)))
    print("full reachable size is {}.".format(len(reachable) + len(terminal)))
    #empowerment for terminal states calculation
    #check if len(terminal) is 0, if it is, set terminal_emp to 0, else set it to the log of the number of unique reachable states
    terminal_emp = np.log2(len(terminal)) if len(terminal) > 0 else 0  # log of the number of unique reachable states
    #testing empowerment alternatives
    emp_ratio = np.log2(len(reachable)-1/(len(terminal) if len(terminal) > 0 else 1))
    print("Empowerment for terminal states from {}: {}".format(state, terminal_emp))
    #empowerment for non-terminal states calculation
    empowerment = np.log2(len(reachable)-1)  # log of the number of unique reachable states
    #complete set of reachable states, terminal or not, would be the "full empowerment"
    full_empowerment = np.log2(len(reachable)+ len(terminal))
    print("Empowerment for non-terminal states from {}: {}".format(state, empowerment))
    print("Sum of both empowerments: {}".format(terminal_emp + empowerment))
    print("Full empowerment for state {}: {}".format(state, full_empowerment))
    print("Empowerment ratio for state {}: {}".format(state, emp_ratio))
       
    return emp_ratio



def deterministic_weighted_empowerment(state, grid, n):
    reachable, terminal = reachable_states_terminal_verbose(state, grid, n)
    reachable_steps, terminal_steps = {}, {}
    gamma = 0.6
    total_decrease = 0
    print("From state: {}".format(state))
    for key, value in reachable.items():
        print("Reachable state: {} at step {}".format(key, value))
        # group the amount of reachable states by the step they were reached
        if value not in reachable_steps:
            reachable_steps[value] = 1
        else:
            reachable_steps[value] += 1
    #print the reachable steps
    for key, value in reachable_steps.items():
        print("Reachable states in step {}: {}".format(key, value))
    #do the same for the terminal states
    for key, value in terminal.items():
        print("Terminal state: {} at step {}".format(key, value))
        if value not in terminal_steps:
            terminal_steps[value] = 1
        else:
            terminal_steps[value] += 1
    for key, value in terminal_steps.items():
        print("Terminal states in step {}: {}".format(key, value))
    #calculate the new empowerment that discounts the terminal empowerment by gamma, and weights the terminal state reachability by a factor of 1/step
    empowerment = np.log2(len(reachable)-1)
    print("Empowerment for non-terminal states from {}: {}".format(state, empowerment))
    for key, value in terminal_steps.items():
        empowerment -= np.log2(value) * gamma / key**2
        total_decrease += np.log2(value) * gamma / key**2
        print("Decreasing the empowerment reduce in step {}: {}".format(key, np.log2(value) * gamma / key**2    ))
    #print the total decrease in empowerment
    print("Total decrease in empowerment: {}".format(total_decrease))
    return empowerment

#rewrite the deterministic_weighted_empowerment function removing the prints 



def normalize_grid(grid):
    grid = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))
    return grid
    
# compute the empowerment for each non-obstacle cell of the grid
#also compute the time taken to compute the empowerment for the whole grid
empowerment_grid = np.zeros(grid.shape)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        if(grid[i, j] != WALL):
            empowerment_grid[i, j] = deterministic_weighted_empowerment((i, j), grid, 2)
        else:
            empowerment_grid[i, j] = 0  # Set to 0 for walls
#print the empowerment grid
print(np.round(empowerment_grid, 4))


#print the grid with the coordinates
# for i in range(grid.shape[0]):
#     for j in range(grid.shape[1]):
#         print("({}, {})".format(i, j), end=" ")
#     print()


# draw the white background onto the surface
windowSurface.fill(WHITE)

#fill the rectangles with white to black gradient according to the empowerment values
#normalize the empowerment values between 0 and 1
empowerment_grid = normalize_grid(empowerment_grid)

#draw all rectangles with the empowerment values
#draw the size proportional to the grid size to make it fit the window according to WINDOWWIDTH and WINDOWHEIGHT values

for i in range(1, grid.shape[0]-1):
    for j in range(1, grid.shape[1]-1):
        pos = (50 + 100 * i, 50 + 100 * j, 100, 100)
        #print the pos for each coordinate formatted as "coordinates: (i, j) pos: (x, y, w, h)"
        #print("coordinates: ({}, {}) pos: {}".format(i, j, pos))
        #calculate color but asserting that values are not lower than 0 or higher than 255
        color = (empowerment_grid[i, j] * 255, empowerment_grid[i, j] * 255, empowerment_grid[i, j] * 255)
        pygame.draw.rect(windowSurface, color, pos)
        pygame.draw.rect(windowSurface, BLACK, pos, 1)
        
# draw walls in the grid as thick black lines in the transitions from the partial obstacles
# if the partial obstacle is from (2, 1) to (2, 2), draw a thick line in the intersection of the two cells
# don't draw diagonal lines
# for partial_obstacle in partial_obstacles:
#     a, b = partial_obstacle
#     length = 100
#     print("Partial obstacle from {} to {}".format(a, b))
#     #if the obstacle is in the same row
#     #draw a line from the top right corner to the bottom right corner of the cell
#     if a[0] == b[0]:
#         x = 100 + 100 * a[0]
#         y = 100 + 100 * a[1]
#         pygame.draw.line(windowSurface, RED, (x + length, y), (x + length, y + length), 5)
#     #if the obstacle is in the same column
#     #draw a line from the bottom left corner to the bottom right corner of the cell
#     elif a[1] == b[1]:
#         x = 100 + 100 * a[0]
#         y = 100 + 100 * a[1]
#         pygame.draw.line(windowSurface, BLACK, (x, y + length), (x + length, y + length), 5)
#     #if the obstacle is diagonal
   
        
# run the game loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    
