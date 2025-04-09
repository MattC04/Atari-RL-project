import torch
import torch.nn as nn

class DQN(nn.Module):
    """Convolutional Neural Network for approximating Q-values with systematic hyperparameter tuning support."""
    
    def __init__(self, input_shape, num_actions, hp_config=None):
        """
        Args:
            input_shape (tuple): Shape of input frames (C, H, W).
            num_actions (int): Number of possible actions in the environment.
            hp_config (dict, optional): Dictionary of hyperparameters to control model architecture.
                Defaults are provided for each key:
                    - 'conv_filters': list of int, number of filters for each conv layer [16, 32]
                    - 'conv_kernel_sizes': list of int, kernel sizes for each conv layer [8, 4]
                    - 'conv_strides': list of int, strides for each conv layer [4, 2]
                    - 'fc_neurons': list of int, neurons for each fully connected layer [256]
                    - 'dropout': float, dropout probability for FC layers (default 0.0, no dropout)
                    - 'activation': str, activation function ('ReLU' or 'LeakyReLU')
        """
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        default_hp = {
            'conv_filters': [16, 32],
            'conv_kernel_sizes': [8, 4],
            'conv_strides': [4, 2],
            'fc_neurons': [256],
            'dropout': 0.0,    
            'activation': 'ReLU'
        }
        #start with defaults otherwise merge 
        if hp_config is None:
            hp_config = {}
        self.hp = {**default_hp, **hp_config}
        
        # Choose activation function based on hyperparameter
        if self.hp['activation'] == 'ReLU':
            activation_fn = nn.ReLU
        elif self.hp['activation'] == 'LeakyReLU':
            activation_fn = nn.LeakyReLU
        else:
            activation_fn = nn.ReLU
        
        # Build convolutional layers dynamically based on hp_config
        conv_layers = []
        in_channels = input_shape[0]
        for out_channels, kernel_size, stride in zip(
            self.hp['conv_filters'], 
            self.hp['conv_kernel_sizes'], 
            self.hp['conv_strides']
        ):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
            conv_layers.append(activation_fn())
            in_channels = out_channels
        
        self.features = nn.Sequential(*conv_layers)
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        convw = input_shape[1]
        convh = input_shape[2]
        for kernel_size, stride in zip(self.hp['conv_kernel_sizes'], self.hp['conv_strides']):
            convw = conv2d_size_out(convw, kernel_size, stride)
            convh = conv2d_size_out(convh, kernel_size, stride)
            
        linear_input_size = convw * convh * self.hp['conv_filters'][-1]
        
        fc_layers = []
        in_features = linear_input_size
        for neurons in self.hp['fc_neurons']:
            fc_layers.append(nn.Linear(in_features, neurons))
            fc_layers.append(activation_fn())
            if self.hp['dropout'] > 0.0:
                fc_layers.append(nn.Dropout(self.hp['dropout']))
            in_features = neurons
            fc_layers.append(nn.Linear(in_features, num_actions))
        
        self.fc = nn.Sequential(*fc_layers)
        
    def forward(self, x):
        x = x / 255.0  # Normalize pixel values to [0, 1]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def count_parameters(self):
        """Returns the number of trainable parameters, which is useful for comparing model variants."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
