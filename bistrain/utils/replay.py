"""
Implementations of Replay Buffers
"""
import random
import torch
import numpy as np
from collections import deque, namedtuple


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """
    def __init__(self, buffer_size, batch_size, seed, device):
        """
        Initialize a ReplayBuffer object.

        Parameters
        ----------
            buffer_size :int
                Maximum size of buffer
            batch_size: int
                Size of each training batch
            seed: int
                Random seed
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # e = self.experience(state, action, reward, next_state, done)
        # self.memory.append(e)
        # In case of multiple agents
        for i in range(state.shape[0]):
            e = self.experience(state[i], action[i], reward[i],
                                next_state[i], done[i])
            self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory
        """
        exps = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in exps
                                  if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in exps
                                   if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in exps
                                   if e is not None])).float().to(self.device)
        n_states = torch.from_numpy(np.vstack([e.next_state for e in exps
                                    if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in exps
                                            if e is not None])
                                 .astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, n_states, dones)

    def __len__(self):
        """
        Return the current size of internal memory
        """
        return len(self.memory)
