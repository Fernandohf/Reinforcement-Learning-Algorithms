"""
Base agent class
"""
from abc import ABC, abstractmethod
import random


class BaseAgent(ABC):
    """
    Base agent class
    """
    def __init__(self, config):
        self.config = config
        super().__init__()

    @abstractmethod
    def step(self):
        """
        Perform one step of the agent learning process.
        """
        pass

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def _learn(self):
        pass


class BaseNoise(ABC):
    """
    Base class for noise processes
    """
    def __init__(self, size, seed):
        self.seed = random.seed(seed)
        self.size = size
        super().__init__()

    @abstractmethod
    def sample(self,):
        pass

    @abstractmethod
    def reset(self):
        pass
