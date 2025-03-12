import gymnasium as gym
import cv2
import numpy as np
from collections import deque

#Overview of file
#Converts raw Atari game frames into a suitable format for NN input
#Basically helps agent recognize important temporal and spatial patterns
#There are two classes: 
# PreprocessFrame(converts RGB grames to grayscale, resizes to 84x84 pixels)
# Framestack(stacks frames together, helps agent perceive movement and dynamics within the environment)
class PreprocessFrame(gym.ObservationWrapper):
    """Converts Atari RGB frames to grayscale and resizes them to the specified shape.
       Note: The provided shape is in (height, width) order.
    """
    def __init__(self, env, shape=(84, 84), grayscale=True):
        super(PreprocessFrame, self).__init__(env)
        self.shape = shape  # desired output shape (height, width)
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                     shape=(1, *self.shape), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                     shape=(3, *self.shape), dtype=np.uint8)
    
    def observation(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        obs = cv2.resize(obs, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            obs = np.expand_dims(obs, axis=0)
        return obs

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, _ = result
        else:
            obs = result
        return self.observation(obs)

class FrameStack(gym.Wrapper):
    """Stacks the last k frames to provide temporal context."""
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                                shape=(k, shp[-2], shp[-1]), dtype=env.observation_space.dtype)
        
    def reset(self):
        result = self.env.reset()
        if isinstance(result, tuple):
            ob = result[0]
        else:
            ob = result
        self.frames.clear()
        frame = np.array(ob, dtype=np.uint8)
        if frame.shape[0] == 1:
            frame = np.squeeze(frame, axis=0)
        for _ in range(self.k):
            self.frames.append(frame)
        return np.stack(self.frames, axis=0)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            ob, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            ob, reward, done, info = result
        frame = np.array(ob, dtype=np.uint8)
        if frame.shape[0] == 1:
            frame = np.squeeze(frame, axis=0)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0), reward, done, info
