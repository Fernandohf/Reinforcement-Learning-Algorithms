"""
Base agent class
"""
from abc import ABC, abstractmethod

from torch.optim import SGD, Adam, AdamW
from torch.nn.utils import clip_grad_norm_
import numpy as np

from ..utils.configuration import BisTrainConfiguration, LocalConfig
from ..networks.actors import FCActorDiscrete, FCActorContinuous
from ..networks.critics import FCCritic, LSTMCritic


class BaseAgent(ABC):
    """
    Base agent class
    """
    def __init__(self, config, noise):
        # Load configuration
        if isinstance(config, str):
            config = BisTrainConfiguration(config)
        elif isinstance(config, dict):
            self.config = LocalConfig(config)
            self._bistrain_config = None
        elif isinstance(config, BisTrainConfiguration):
            pass
        else:
            raise ValueError("Configuration file is invalid!")

        if isinstance(config, BisTrainConfiguration):
            # Main configuration
            self._bistrain_config = config
            agent = config["AGENT"].upper()
            self.config = LocalConfig(config[agent])

        # Noise process
        self.noise = noise

        super().__init__()

    # def _set_noise(self):
    #     config = self.config.EXPLORATION
    #     if config.TYPE == 'ou':
    #         noise = OUNoise(config)
    #     else:
    #         # Default
    #         noise = GaussianNoise(config)
    #     return noise

    def _set_policy(self, optimizer=True):
        """
        Creates the network defined in the [ACTOR] subsection
        of the configuration file.

        Parameters
        ----------
        opmizer: bool
            Wether define the optimizer or not
        """
        actor_config = self.config.ACTOR
        # FC architecture
        if actor_config.ARCHITECTURE == 'fc':
            # Continuous
            if self.config.ACTION_SPACE == 'continuous':
                policy = FCActorContinuous(self.config.STATE_SIZE,
                                           self.config.ACTION_SIZE,
                                           actor_config.HIDDEN_SIZE,
                                           self.config.SEED,
                                           actor_config.HIDDEN_ACTIV,
                                           actor_config.OUTPUT_LOC_ACTIV,
                                           actor_config.OUTPUT_SCALE_ACTIV,
                                           actor_config.OUTPUT_LOC_SCALER,
                                           self.config.ACTION_RANGE)
            # Discrete
            elif self.config.ACTION_SPACE == 'discrete':
                policy = FCActorDiscrete(self.config.STATE_SIZE,
                                         self.config.ACTION_SIZE,
                                         actor_config.HIDDEN_SIZE,
                                         self.config.SEED,
                                         actor_config.HIDDEN_ACTIV)
        # TODO LSTM architecture ACTORS
        # Move to device
        policy = policy.to(self.config.DEVICE)
        # Add optimizer
        if optimizer:
            policy.__setattr__("optimizer",
                               self._set_optimizer(policy.parameters(),
                                                   actor_config))

        return policy

    def _set_val_func(self, optimizer=True):
        """
        Creates the network defined in the [CRITIC] subsection
        of the configuration file.

        Parameters
        ----------
        opmizer: bool
            Wether define the optimizer or not
        """
        critic_config = self.config.CRITIC
        if critic_config.ARCHITECTURE == 'fc':
            val_func = FCCritic(self.config.STATE_SIZE,
                                self.config.ACTION_SIZE,
                                critic_config.HIDDEN_SIZE,
                                self.config.SEED)
        elif critic_config.ARCHITECTURE == 'lstm':
            val_func = LSTMCritic(self.config.STATE_SIZE,
                                  self.config.ACTION_SIZE,
                                  critic_config.HIDDEN_SIZE,
                                  self.config.SEED)
        # Move to device and return
        val_func = val_func.to(self.config.DEVICE)
        # Add optimizer
        if optimizer:
            val_func.__setattr__("optimizer",
                                 self._set_optimizer(val_func.parameters(),
                                                     critic_config))

        return val_func

    def _set_optimizer(self, parameters, config):
        if config.OPTIMIZER == 'sgd':
            optimizer = SGD(parameters, lr=config.LR,
                            momentum=config.MOMENTUM)
        elif config.OPTIMIZER == 'adamw':
            optimizer = AdamW(parameters, lr=config.LR,
                              weight_decay=config.WEIGHT_DECAY)
        else:
            # Default
            optimizer = Adam(parameters, lr=config.LR,
                             weight_decay=config.WEIGHT_DECAY)
        return optimizer

    def _add_noise(self, action):
        """
        Modify the action with noise
        """
        # Continuous actions
        if self.config.ACTION_SPACE == "continuous":
            action += self.noise.sample()
            # Clipped action
            action = np.clip(action,
                             *self.config.ACTION_RANGE)
        # Discrete Actions
        elif self.config.ACTION_SPACE == "discrete":
            action = self.noise.sample(action)
        return action

    def _clip_gradient(self, model):
        """
        Perform graient clipping in given model
        """
        # Gradient clipping
        if self.config.TRAINING.GRADIENT_CLIP != 0:
            clip_grad_norm_(model.parameters(),
                            self.config.TRAINING.GRADIENT_CLIP)

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
