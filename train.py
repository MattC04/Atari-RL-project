import gymnasium as gym
import torch
import torch.optim as optim
import random
import cv2
from dqn import DQN
from replay_memory import ReplayMemory
from environment import PreprocessFrame, FrameStack
from utils import optimize_model

def train_dqn(env_name, num_episodes=100000):
    env = gym.make(env_name)
    env = PreprocessFrame(env, shape=(84, 84))
    env = FrameStack(env, 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = ReplayMemory(100000)

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = random.randrange(env.action_space.n)
            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            optimize_model(policy_net, target_net, memory, optimizer, 32, 0.99, device)

        print(f"Episode {episode} | Reward: {total_reward}")

        if episode % 1000 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return policy_net, target_net, [], []
