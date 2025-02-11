
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

gym.register_envs(gymnasium_robotics)

policy_save_path = "/Trained policies/"

TEST = "test"
TRAIN = "train"
EVAL = "eval"
HUMAN = "human"
RGB = "rgb_array"
G = "g"
R = "r"

emp_test_maze = [ 
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, G, G, 0, 0, 0, 0, 1],
    [1, G, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, R, R, R, R, R, R, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]

def setup_and_test_env():
    # Initialize the PointMaze environment with U_MAZE configuration
    env = gym.make('PointMaze_UMazeDense-v3', maze_map=emp_test_maze, render_mode='human')
    # Reset the environment to get the initial observation
    observation, info = env.reset(seed=42)  # Seed ensures reproducibility

    done = False
    total_reward = 0

    while not done:
        env.render()  # Optional: Render the environment visualization
        
        # Select a random action (replace with your RL policy later)
        action = env.action_space.sample()
        
        # Take a step in the environment
        observation, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Print observations and rewards
        print(f"Action: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        print("---")

    print(f"Total reward for the episode: {total_reward}")
    env.close()

def train_and_test_agent():
    # Wrap the environment for vectorized training
    env = make_vec_env(
        lambda: gym.make(
            'PointMaze_UMazeDense-v3', render_mode='rgb_array'
        ), 
        n_envs=1
    )

    # Train a PPO agent
    print("Starting training...")
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    #save the model for evaluation later
    model.save("ppo_pointmaze-emp")
    print("Model saved as ppo_pointmaze-emp.zip.")

    # Test the trained agent
    print("Starting evaluation...")
    eval_env = gym.make('PointMaze_UMazeDense-v3', render_mode='human', width=1200, height=1200)
    obs, info = eval_env.reset()
    for _ in range(10000):  # Fixed number of evaluation steps
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = eval_env.step(action)
        eval_env.render()
        if done or truncated:
            obs, info = eval_env.reset()
    eval_env.close()
    
def train_agent_save(filename, render):
    # Wrap the environment for vectorized training
    env = make_vec_env(
        lambda: gym.make(
            'PointMaze_UMazeDense-v3', maze_map=emp_test_maze, render_mode=render, width=1200, height=1200
        ), 
        n_envs=1
    )

    # Train a PPO agent
    print("Starting training...")
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=400000)
    #save the model for evaluation later
    model.save(filename)

def evaluate_saved_model(filename, render):
    # Load the saved model
    model = PPO.load(filename)
    print(f"Model loaded from {filename}.")

    # Evaluate the model
    eval_env = gym.make('PointMaze_UMazeDense-v3', maze_map=emp_test_maze, render_mode=render, width=1200, height=1200)
    obs, info = eval_env.reset()
    for _ in range(10000):  # Fixed number of evaluation steps
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = eval_env.step(action)
        eval_env.render()
        if done or truncated:
            obs, info = eval_env.reset()
    eval_env.close()

if __name__ == "__main__":
    
    mode = EVAL
    filename = "ppo_pointmaze-reward"
    render = HUMAN
    
    if(mode == TEST):
        print("Testing the environment...")
        setup_and_test_env()
    if(mode == TRAIN):       
        print("Training the RL agent...")        
        train_agent_save(filename, render)
    if(mode == EVAL):
        print("Evaluating the trained agent...")        
        print("Evaluating model saved as ", filename)
        evaluate_saved_model(filename, render)