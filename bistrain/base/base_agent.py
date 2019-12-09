"""
Base agent class
"""
from abc import ABC, abstractmethod

from torch.optim import SGD, Adam, AdamW

from ..utils.configuration import BisTrainConfiguration
from ..utils.noise import GaussianNoise, OUNoise


class BaseAgent(ABC):
    """
    Base agent class
    """

    def __init__(self, config_file):
        # Load configuration
        self.config = BisTrainConfiguration(config_file)
        self.config.activate_sections("AGENT")
        super().__init__()

    def _set_optimizer(self, parameters):
        if self.config.OPTIMIZER == 'sgd':
            optimizer = SGD(parameters, lr=self.config.LR,
                            momentum=self.config.MOMENTUM)
        elif self.config.OPTIMIZER == 'adamw':
            optimizer = AdamW(parameters, lr=self.config.LR,
                              weight_decay=self.config.WEIGHT_DECAY)
        else:
            # Default
            optimizer = Adam(parameters, lr=self.config.LR,
                             weight_decay=self.config.WEIGHT_DECAY)
        return optimizer

    def _set_noise(self):
        self.config.activate_sections("EXPLORATION")
        if self.config.TYPE == 'ou':
            noise = OUNoise(self.config)
        else:
            # Default
            noise = GaussianNoise(self.config)
        self.config.activate_sections("AGENT")
        return noise

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
