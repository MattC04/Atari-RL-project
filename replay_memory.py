### replay_memory.py - Implements Experience Replay

import random
import numpy as np
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        """Initialize replay memory with a fixed capacity."""
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        """Save a transition into memory."""
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        """Retrieve a random batch of transitions."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
