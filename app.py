### app.py - Main script to run training and evaluation.

import gymnasium as gym
import torch
from train import train_dqn
from dqn import DQN
from preprocessing import PreprocessFrame, FrameStack


def main():
    env_name = "ALE/Assault-v5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment
    env = gym.make(env_name)
    env = PreprocessFrame(env, shape=(84, 84))
    env = FrameStack(env, 4)

    # Train the model
    train_dqn(env_name, device)

    # Load trained model for evaluation
    model = DQN(env.observation_space.shape, env.action_space.n).to(device)
    model.load_state_dict(torch.load("dqn_model.pth"))
    model.eval()

    print("Training complete. Evaluating model...")

    # Evaluate the trained model
    from train import evaluate_policy
    evaluate_policy(model, env, num_episodes=3, device=device, render=True)


if __name__ == "__main__":
    main()
