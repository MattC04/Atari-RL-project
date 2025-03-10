import torch
import gymnasium as gym
from train import train_dqn
from evaluate import evaluate_policy
from environment import PreprocessFrame, FrameStack

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")

    env_name = "ALE/Assault-v5"
    verification_env = gym.make(env_name)
    print("Action meanings:", verification_env.unwrapped.get_action_meanings())
    verification_env.close()

    policy_net, target_net, losses, episode_rewards = train_dqn(env_name)

    torch.save(policy_net.state_dict(), "dqn_assault.pth")

    render_env = gym.make(env_name, render_mode='human')
    render_env = PreprocessFrame(render_env, shape=(84, 84))
    render_env = FrameStack(render_env, 4)

    evaluate_policy(policy_net, render_env, num_episodes=3, device=device, render=True)
