"""
Base noise class
"""
from abc import ABC, abstractmethod

from ..utils.configuration import LocalConfig


class BaseNoise(ABC):
    """
    Base class for noise processes
    """

    def __init__(self, config):
        self.config = LocalConfig(config)
        super().__init__()

    @abstractmethod
    def sample(self,):
        pass

    @abstractmethod
    def reset(self):
        pass
