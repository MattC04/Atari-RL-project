import torch
import time

def evaluate_policy(policy_net, env, num_episodes=10, device=torch.device("cpu"), render=False):
    for i in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action = policy_net(torch.tensor(state).float().unsqueeze(0).to(device)).argmax().item()
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if render:
                env.render()
                time.sleep(0.02)

        print(f"Evaluation Episode {i} | Reward: {total_reward}")
