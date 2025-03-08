import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import os
from dqn import DQN, get_optimizer


def train_dqn(env_name, num_episodes=100000, batch_size=32, gamma=0.99, learning_rate=1e-4, target_update=1000, epsilon_decay=1000000):
    env = gym.make(env_name)
    num_actions = env.action_space.n
    input_shape = env.observation_space.shape
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(input_shape, num_actions).to(device)
    target_net = DQN(input_shape, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = get_optimizer(policy_net, learning_rate)
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            epsilon = max(0.1, 1 - episode / epsilon_decay)
            action = env.action_space.sample() if np.random.rand() < epsilon else policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)).argmax().item()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
        
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        print(f"Episode {episode}: Reward {total_reward}")
    
    torch.save(policy_net.state_dict(), "dqn_model.pth")
