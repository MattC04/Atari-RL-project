import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 9 * 9, 256),  # Adjust based on input shape
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def get_optimizer(policy_net, learning_rate=1e-4):
    return optim.RMSprop(policy_net.parameters(), lr=learning_rate)
