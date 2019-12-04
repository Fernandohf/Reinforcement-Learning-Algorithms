"""
Base agent class
"""
from abc import ABC, abstractmethod
import random
from .load.configuration import BisTrainConfiguration
from torch.optim import Adam, AdamW, SGD


class BaseAgent(ABC):
    """
    Base agent class
    """
    def __init__(self, config_file):
        # Load configuration
        self.config = BisTrainConfiguration(config_file)
        self.config.activate_section("AGENT")
        super().__init__()

    def _set_optimizer(self, config_value, parameters, lr, weigh_decay, momentum):
        if config_value == 'sgd':
            return SGD(parameters, lr=lr, momentum=momentum)
        elif config_value == 'adamw':
            return AdamW(parameters, lr=lr, weight_decay=weigh_decay)
        else:
            # Default
            return Adam(parameters, lr=lr, weight_decay=weigh_decay)

    def _set_noise(self, config_value, config):
        if config_value == 'sgd':
            return SGD(parameters, lr=lr, momentum=momentum)
        elif config_value == 'adamw':
            return AdamW(parameters, lr=lr, weight_decay=weigh_decay)
        else:
            # Default
            return Adam(parameters, lr=lr, weight_decay=weigh_decay)

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
    def __init__(self, config):
        self.config = config
        self.config.activate_section("EXPLORATION")
        super().__init__()

    @abstractmethod
    def sample(self,):
        pass

    @abstractmethod
    def reset(self):
        pass
