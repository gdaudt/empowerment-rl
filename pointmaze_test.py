
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

gym.register_envs(gymnasium_robotics)

def setup_and_test_env():
    # Initialize the PointMaze environment with U_MAZE configuration
    env = gym.make('PointMaze_UMazeDense-v3', render_mode='human')
    
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

    # Test the trained agent
    print("Starting evaluation...")
    eval_env = gym.make('PointMaze_UMazeDense-v3', render_mode='human')
    obs, info = eval_env.reset()
    for _ in range(10000):  # Fixed number of evaluation steps
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = eval_env.step(action)
        eval_env.render()
        if done or truncated:
            obs, info = eval_env.reset()
    eval_env.close()

if __name__ == "__main__":
    # print("Testing the environment...")
    # setup_and_test_env()
    print("Training and testing the RL agent...")
    train_and_test_agent()