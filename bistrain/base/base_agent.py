"""
Base agent class
"""
from abc import ABC, abstractmethod

from torch.optim import SGD, Adam, AdamW

from ..utils.configuration import BisTrainConfiguration
from ..utils.noise import GaussianNoise, OUNoise
from ..networks.actors import FCActorDiscrete, FCActorContinuous
from ..networks.critics import FCCritic, LSTMCritic


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
        self.config.deactivate_subsection()
        return noise

    def _set_policy(self):
        self.config.activate_subsection("ACTOR")
        # FC architecture
        if self.config.ARCHITECTURE == 'fc':
            if self.config.ACTION_SPACE == 'continuous':
                policy = FCActorContinuous(self.config.STATE_SIZE,
                                           self.config.ACTION_SIZE,
                                           tuple(self.config.HIDDEN_SIZE),
                                           self.config.SEED).to(self.config
                                                                .DEVICE)
            elif self.config.ACTION_SPACE == 'discrete':
                policy = FCActorDiscrete(self.config.STATE_SIZE,
                                         self.config.ACTION_SIZE,
                                         tuple(self.config.HIDDEN_SIZE),
                                         self.config.SEED).to(self.config
                                                              .DEVICE)
        # Deactivate subsection
        self.config.deactivate_subsection()
        return policy

    def _set_val_func(self):
        self.config.activate_subsection("CRITIC")
        if self.config.ARCHITECTURE == 'fc':
            val_func = FCCritic(self.config.STATE_SIZE,
                                self.config.ACTION_SIZE,
                                tuple(self.config.HIDDEN_SIZE),
                                self.config.SEED).to(self.config.DEVICE)
        elif self.config.ARCHITECTURE == 'lstm':
            val_func = LSTMCritic(self.config.STATE_SIZE,
                                  self.config.ACTION_SIZE,
                                  tuple(self.config.HIDDEN_SIZE),
                                  self.config.SEED).to(self.config.DEVICE)
        # Deactivate subsection
        self.config.deactivate_subsection()
        return val_func

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
