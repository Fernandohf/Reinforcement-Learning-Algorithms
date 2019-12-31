"""
Implementations of Replay Buffers
"""
import random
from collections import deque, namedtuple

import torch
import numpy as np

from ..utils.configuration import LocalConfig


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """
    def __init__(self, config):
        """
        Initialize a ReplayBuffer object.

        Parameters
        ----------
            config: dict or LocalConfig
                All configuration parameters of the buffer
                (buffer_size, batch_size, seed, device)
        """
        self.config = LocalConfig(config)
        # Interal memmory
        self.memory = deque(maxlen=int(self.config.BUFFER_SIZE))
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        shape_len = len(state.shape)
        # In case of multiple agents
        if shape_len > 2:
            for i in range(state.shape[0]):
                e = self.experience(state[i], action[i], reward[i],
                                    next_state[i], done[i])
                self.memory.append(e)
        # Single agent
        else:
            e = self.experience(state, action, reward,
                                next_state, done)
            self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory
        """
        exps = random.sample(self.memory, k=self.config.BATCH_SIZE)

        states = (torch.from_numpy(np.vstack([e.state for e in exps
                  if e is not None])).float().to(self.config.DEVICE))
        actions = (torch.from_numpy(np.vstack([e.action for e in exps
                   if e is not None])).float().to(self.config.DEVICE))
        rewards = (torch.from_numpy(np.vstack([e.reward for e in exps
                   if e is not None])).float().to(self.config.DEVICE))
        n_states = (torch.from_numpy(np.vstack([e.next_state for e in exps
                    if e is not None])).float().to(self.config.DEVICE))
        dones = (torch.from_numpy(np.vstack([e.done for e in exps
                 if e is not None]).astype(np.uint8))
                 .float().to(self.config.DEVICE))

        return (states, actions, rewards, n_states, dones)

    def __len__(self):
        """
        Return the current size of internal memory
        """
        return len(self.memory)
