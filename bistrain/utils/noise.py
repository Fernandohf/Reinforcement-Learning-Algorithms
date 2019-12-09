"""
Noise processes to add to actions
"""
import random
from copy import copy

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

        self.reset()

    def reset(self):
        """
        Reset the internal decay status to starting values
        """
        # Reset Epsilon
        self._eps = self.config.EPS_BETA
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
            [np.exp(-self.config.EPS_BETA * self._eps_step),
             self.config.EPS_MIN])
        self._eps_step += 1

        # Sample
        return np.random.normal(loc=self.config.MEAN,
                                scale=self.config.SIGMA,
                                size=self.config.SIZE) * self._eps


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
        self.state = copy(self.config.MEAN)

    def sample(self):
        """
        Update internal state and return it as a noise sample

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
