import gymnasium as gym
import time
import torch
import numpy as np

def evaluate_policy(policy_net, env, num_episodes=10, device=torch.device("cpu"), render=False):
    rewards = []
    for i in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        if render:
            env.unwrapped.render_mode = 'human'
            env.unwrapped.render()
            
        while not done:
            state_tensor = torch.from_numpy(np.expand_dims(state, 0)).float().to(device)
            with torch.no_grad():
                action = policy_net(state_tensor).max(1)[1].item()
                
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            if render:
                env.unwrapped.render()
                time.sleep(0.02)
            
            state = next_state
            
        rewards.append(total_reward)
        print(f"Evaluation Episode {i} | Reward: {total_reward}")
    
    if render:
        env.unwrapped.close()
        
    return rewards
