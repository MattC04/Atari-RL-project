import random
from collections import deque, namedtuple

#Overview of file
#Implements simple memory buffer to manage memory through double ended queue
#Defines structured tuples that represent (state, action, reward, nextstate, whether its done or not)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
