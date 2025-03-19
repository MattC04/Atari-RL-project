# 🎮 Atari Reinforcement Learning 

## Motivation  
Reinforcement Learning has been gaining a lot of traction as a powerful approach to train AI agents for complex decision-making tasks. This project applies RL to the classic Atari Assault game, leveraging deep learning techniques to train an AI to play autonomously. The goal is to explore how RL can learn game mechanics, improve performance over time, and adapt strategies to maximize rewards.

## Project Overview  
This project implements an RL agent using **Deep Q-Networks (DQN)** to train an AI for playing Atari games efficiently. The environment is simulated using **OpenAI Gym's Atari framework**, and the model is trained using PyTorch.

### What We Did  
- **Preprocessed the Game Environment:**  
  - Converted frames to grayscale and resized them to 84x84 pixels for efficiency.  
  - Stacked frames to provide temporal context.  
  - Applied frame skipping to speed up learning.  

- **Built the RL Agent:**  
  - Implemented a Deep Q-Network (DQN) with experience replay.  
  - Used **Convolutional Neural Networks (CNNs)** for feature extraction.  
  - Tuned hyperparameters like learning rate, discount factor, and exploration strategy.  

- **Trained the Agent:**  
  - Used **epsilon-greedy exploration** to balance exploration and exploitation.  
  - Stored experiences in a replay buffer and trained the model in batches.  
  - Implemented a **control mechanism** to manually stop training.  
  - Checkpoints were saved to resume training efficiently.  

- **Visualized and Analyzed Results:**  
  - Tracked training performance metrics.  
  - Rendered gameplay to assess learned behavior.  
  - Compared performance across different RL models.  

## How It Works  

### Model Architecture  
- The **DQN model** consists of:  
  - **Two convolutional layers** for extracting spatial features from game frames.  
  - **Fully connected layers** to map extracted features to Q-values (expected rewards for each possible action).  
  - **ReLU activation functions** for non-linearity and improved learning.  

### Training Process  
1. The RL agent **observes** the game environment, processes visual input, and **chooses actions** to maximize cumulative rewards.  
2. A **neural network** predicts action values (Q-values), learning from past experiences via **experience replay**.  
3. The agent trains over multiple episodes, progressively improving its gameplay.  
4. The **epsilon-greedy strategy** balances exploration and exploitation.  
5. **Checkpoints** allow training to resume from the last saved state.  

### Key Components  
- **[`train.py`](train.py)**: Implements the training loop, policy updates, and checkpointing.  
- **[`main.py`](main.py)**: Entry point that initializes and coordinates training/evaluation.  
- **[`model.py`](model.py)**: Defines the **Deep Q-Network (DQN) architecture** using CNNs.  
- **[`replay.py`](replay.py)**: Implements **experience replay** for stable training.  
- **[`preprocess.py`](preprocess.py)**: Handles **frame preprocessing** (grayscale conversion, resizing, stacking).  
- **[`evaluate.py`](evaluate.py)**: Evaluates trained models by running multiple episodes.  
- **[`control.py`](control.py)**: Implements a manual **stop training** mechanism using OpenCV.  

## Outcome  
- The trained agent demonstrates **significant improvement** in gameplay over time.  
- Experimentation with different RL algorithms showcases their impact on learning efficiency.  
- Performance varies based on hyperparameters, training duration, and game complexity.  

## Gameplay Video  
[![Watch the gameplay]
<video src="Playing-Atari-with-Reinforcement-Learning.mp4" controls width="600"></video>

---

## Installation & Setup  
To run this project locally, follow these steps:

### Prerequisites  
Ensure you have the following installed:  
- Python 3.x  
- OpenAI Gym  
- PyTorch  
- NumPy, Matplotlib  
- `ale-py` for Atari environments  

### Installation  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/MattC04/Atari-RL-project.git
cd Atari-RL-project
pip install -r requirements.txt
```

### Running the RL Agent  

#### Training  
```bash
python main.py  
```
This will start training the RL agent with DQN and save model checkpoints.  

#### Evaluating the Trained Model  
```bash
python evaluate.py  
```
This will run the trained agent and display performance metrics.  

## Future Improvements  
- Experiment with **more complex RL algorithms** (e.g., PPO, A3C).  
- Improve computational efficiency with **better hardware acceleration**.  
- Extend to **multi-agent RL** for competitive gameplay.  

---

## Contributors  

- **[MattC04](https://github.com/MattC04)** - Lead Developer  

---

