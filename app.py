import torch
import gymnasium as gym
import os
from dqn import DQN
from train import train_dqn

if __name__ == "__main__":
    env_name = "ALE/Assault-v5"
    checkpoint_path = "dqn_model.pth"
    
    if os.path.exists(checkpoint_path):
        print("Loading trained model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = gym.make(env_name)
        input_shape = env.observation_space.shape
        num_actions = env.action_space.n
        policy_net = DQN(input_shape, num_actions).to(device)
        policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        policy_net.eval()
        print("Model loaded successfully.")
    else:
        print("Training model from scratch...")
        train_dqn(env_name)
    
    print("Model ready for evaluation or further training.")
