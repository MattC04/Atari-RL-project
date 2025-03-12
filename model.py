import torch
import torch.nn as nn

#Overview of file
#Defines the DQN architecture using CNN designed for Atari gameplay
#processes the game input frames through convulational layers to extract meaningful features
#Two convulational layers (first layer:16 filters, Second layer: 32 filters)
#Computes map dimensions to connect the convolutional output to the fully connected layers
#fully connected layers interpret extracted visual features and maps them to Q-values (expected rewards for each possible action)

class DQN(nn.Module):
    """Convolutional Neural Network for approximating Q-values."""
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2)
        convh = conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2)
        linear_input_size = convw * convh * 32
        
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
    def forward(self, x):
        x = x / 255.0  # normalize pixel values
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
