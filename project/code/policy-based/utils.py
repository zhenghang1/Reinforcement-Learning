import random
from typing import Any
import numpy as np
import torch
from collections import deque, namedtuple


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, seed, discount_factor, n_step=1):
        self.device = device
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.discount_factor = discount_factor
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer)
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones

    def calc_multistep_return(self, n_step_buffer):
        return_val = 0
        for i in range(self.n_step):
            return_val += self.discount_factor ** i * n_step_buffer[i][2]

        return n_step_buffer[0][0], n_step_buffer[0][1], return_val, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def __len__(self):
        return len(self.memory)
    
class ReplayBuffer_prob:
    def __init__(self, buffer_size, batch_size, device, seed, discount_factor, n_step=1):
        self.device = device
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "log_prob"])
        self.discount_factor = discount_factor
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        random.seed(seed)

    def add(self, state, action, reward, next_state, done, log_prob):
        self.n_step_buffer.append((state, action, reward, next_state, done, log_prob))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done, log_prob = self.calc_multistep_return(self.n_step_buffer)
            e = self.experience(state, action, reward, next_state, done, log_prob)
            self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)
        log_probs = torch.from_numpy(np.vstack([e.log_prob for e in experiences if e is not None])).float().to(
            self.device)

        return states, actions, rewards, next_states, dones, log_probs

    def calc_multistep_return(self, n_step_buffer):
        return_val = 0
        for i in range(self.n_step):
            return_val += self.discount_factor ** i * n_step_buffer[i][2]

        return n_step_buffer[0][0], n_step_buffer[0][1], return_val, n_step_buffer[-1][3], n_step_buffer[-1][4], n_step_buffer[0][5]

    def __len__(self):
        return len(self.memory)
    

def get_device(cuda):
    if cuda < 0:
        print('Using the CPU.',flush=True)
        device = torch.device("cpu")
    elif not torch.cuda.is_available():
        print('No GPU available, using the CPU instead.',flush=True)
        device = torch.device("cpu")
    else:
        gpu_count = torch.cuda.device_count()
        print(f'There are {gpu_count} GPU(s) available.',flush=True)
        if gpu_count < cuda:
            print(f'Cuda:{cuda} is not available, using cuda:{0} instead.',flush=True)
            device = torch.device(f"cuda:{0}")
            print('Device name:', torch.cuda.get_device_name(0),flush=True)
        else:
            print(f'Using cuda:{cuda}',flush=True)
            device = torch.device(f"cuda:{cuda}")
            print('Device name:', torch.cuda.get_device_name(cuda),flush=True)
    return device


class RunningMeanStd(object):
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.stdd = np.ones(shape, 'float64')
        self.count = 0

    def __call__(self, state):
        self.count += 1
        self.update(state)

        state = (state - self.mean) / (self.std + 1e-8)
        state = np.clip(state, -5, +5)

        return state

    def update(self,state):
        if self.count == 1:
            self.mean = state
        else:
            old_mean = self.mean
            self.mean = old_mean + (state - old_mean) / self.count
            self.stdd = self.stdd + (state - old_mean) * (state - self.mean)

        if self.count > 1:
            self.std = np.sqrt(self.stdd / (self.count - 1))




""""
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code Expanded and Adapted from Code provided by Udacity DRL Team, 2018.
"""

import copy
import random
import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.1, sigma=.1, sigma_min = 0.05, sigma_decay=.99):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        """Resduce  sigma from initial value to min"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state