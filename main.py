import os
import torch
import cv2
import gymnasium as gym

import control  # sets up the control window and global stop_training
from train import train_dqn
from evaluate import evaluate_policy
from preprocess import PreprocessFrame, FrameStack

if __name__ == "__main__":
    import ale_py  # Required for ALE environments
    
    # 1. Verify environment actions
    env_name = "ALE/Assault-v5"
    verification_env = gym.make(env_name)
    print("Action meanings:", verification_env.unwrapped.get_action_meanings())
    verification_env.close()
    
    # 2. Check for an existing checkpoint
    checkpoint_path = "dqn_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        start_episode = checkpoint["episode"]
        steps_done = checkpoint["steps_done"]
        losses = checkpoint["losses"]
        episode_rewards = checkpoint["episode_rewards"]
        memory = checkpoint["memory"]
        resume_policy_net_state = checkpoint["policy_net_state_dict"]
        resume_target_net_state = checkpoint["target_net_state_dict"]
        # Do not resume optimizer state since we're switching to Adam
        resume_optimizer_state = None
        print(f"Resuming training from episode {start_episode}")
    else:
        start_episode = 0
        steps_done = 0
        losses = []
        episode_rewards = []
        memory = None
        resume_policy_net_state = None
        resume_target_net_state = None
        resume_optimizer_state = None

    # 3. Training phase (this will resume if a checkpoint was found)
    policy_net, target_net, losses, episode_rewards = train_dqn(
        env_name,
        num_episodes=100000,
        target_update=10000,
        epsilon_decay=1000000,
        start_episode=start_episode,
        steps_done=steps_done,
        losses=losses,
        episode_rewards=episode_rewards,
        memory=memory,
        resume_policy_net_state=resume_policy_net_state,
        resume_target_net_state=resume_target_net_state,
        resume_optimizer_state=resume_optimizer_state
    )
    
    # 4. Save the trained model (if training finished normally)
    torch.save(policy_net.state_dict(), "dqn_assault.pth")

    # Destroy the control window after training
    cv2.destroyWindow("Control")

    # 5. Evaluation with rendering (using 84x84 window size)
    render_env = gym.make(env_name, render_mode='human')
    render_env = PreprocessFrame(render_env, shape=(84, 84))
    render_env = FrameStack(render_env, 4)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    evaluate_policy(policy_net, render_env, num_episodes=3, device=device, render=True)
