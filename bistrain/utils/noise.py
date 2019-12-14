"""
Noise processes to add to actions
"""
import random

import numpy as np

from ..base.base_noise import BaseNoise


class GaussianNoise(BaseNoise):
    """
    Normal noise with decay generator.
    """

    def __init__(self, config):
        """
        Initialize parameters and noise process

        Parameters
        ----------
        config: BisTrainConiguration
            Configuration of the noise
        """
        # Base class initialization
        super().__init__(config)
        # Parameters
        self.eps_beta = self.config["EXPLORATION"]["EPS_BETA"]
        self.mean = self.config["EXPLORATION"]["MEAN"]
        self.std = self.config["EXPLORATION"]["SIGMA"]
        self.size = self.config["GLOBAL"]["ACTION_SIZE"]
        self.eps_min = self.config["EXPLORATION"]["EPS_MIN"]
        self.reset()

    def reset(self):
        """
        Reset the internal decay status to starting values
        """
        # Reset Epsilon
        self._eps = self.eps_beta
        self._eps_step = 0

    def sample(self):
        """
        Update internal decay rate and return a decayed noise sample

        Return
        ------
         :float
            Decayed noise sample
        """
        # Update epsilon
        self._eps = max(
            [np.exp(-self.eps_beta * self._eps_step),
             self.eps_min])
        self._eps_step += 1

        # Sample
        return np.random.normal(loc=self.mean,
                                scale=self.std,
                                size=self.size) * self._eps


class OUNoise(BaseNoise):
    """
    Ornstein-Uhlenbeck process also called Damped Random Walk (DRW).
    """

    def __init__(self, config):
        """
        Initialize parameters and noise process

        Parameters
        ----------
        config: BisTrainConiguration
            Configuration of the noise
        """
        # Base class initialization
        super().__init__(config)

        self.reset()

    def reset(self):
        """
        Reset the internal state to mean
        """
        self.state = np.ones(self.config.ACTION_SIZE) * self.config.MEAN

    def sample(self):
        """
        Update internal state sand return it as a noise sample

        Return
        ------
        state: float
            Sample from the process
        """
        x = self.state
        dx = self.config.THETA * (self.config.MEAN - x) + self.config.SIGMA * \
            np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
