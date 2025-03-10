import gymnasium as gym
import cv2
import numpy as np
from collections import deque

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84), grayscale=True):
        super(PreprocessFrame, self).__init__(env)
        self.shape = shape
        self.grayscale = grayscale

    def observation(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = np.expand_dims(obs, axis=0)
        return cv2.resize(obs, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        self.frames.append(state)
        return np.stack(self.frames, axis=0), reward, done
