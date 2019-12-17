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
    def __init__(self, config):
        # Load configuration
        if isinstance(config, str):
            config = BisTrainConfiguration(config)
        elif isinstance(config, BisTrainConfiguration):
            config = config
        else:
            raise ValueError("Configuration file is invalid!")

        # Extract agent parameters
        agent = config["GLOBAL"]["AGENT"].upper()
        self.con

        # Parameters
        self.optimizer

        super().__init__()

    def _set_optimizer(self, parameters):
        if config.OPTIMIZER == 'sgd':
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
        if self.config["EXPLORATION"]["TYPE"] == 'ou':
            noise = OUNoise(self.config)
        else:
            # Default
            noise = GaussianNoise(self.config)
        return noise

    def _set_policy(self):
        # FC architecture
        if self.config.ARCHITECTURE == 'fc':
            # Continuous
            if self.config.ACTION_SPACE == 'continuous':
                policy = FCActorContinuous(self.config.STATE_SIZE,
                                           self.config.ACTION_SIZE,
                                           self.config.HIDDEN_SIZE,
                                           self.config.SEED,
                                           self.config.HIDDEN_ACTIV,
                                           self.config.OUTPUT_LOC_ACTIV,
                                           self.config.OUTPUT_SCALE_ACTIV,
                                           self.config.OUTPUT_LOC_SCALER,
                                           self.config.ACTION_RANGE)
            # Discrete
            elif self.config.ACTION_SPACE == 'discrete':
                policy = FCActorDiscrete(self.config.STATE_SIZE,
                                         self.config.ACTION_SIZE,
                                         self.config.HIDDEN_SIZE,
                                         self.config.SEED,
                                         self.config.HIDDEN_ACTIV)
        # TODO LSTM architecture ACTORS
        return policy.to(self.config.DEVICE)

    def _set_val_func(self):
        if self.config.ARCHITECTURE == 'fc':
            val_func = FCCritic(self.config.STATE_SIZE,
                                self.config.ACTION_SIZE,
                                self.config.HIDDEN_SIZE,
                                self.config.SEED)
        elif self.config.ARCHITECTURE == 'lstm':
            val_func = LSTMCritic(self.config.STATE_SIZE,
                                  self.config.ACTION_SIZE,
                                  self.config.HIDDEN_SIZE,
                                  self.config.SEED)
        # Move to device and return
        return val_func.to(self.config.DEVICE)

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
