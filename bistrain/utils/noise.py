"""
Noise processes to add to actions
"""
import random
import numpy as np
from copy import copy
from ..base import BaseNoise


class GaussianNoise(BaseNoise):
    """
    Normal noise with decay generator.
    """

    def __init__(self, size, seed, mu=0., sigma=.2,
                 eps_beta=.01, eps_init=1., eps_min=.001):
        """
        Initialize parameters and noise process

        Parameters
        ----------
        mu: float
            Mean value
        sigma: float
            Standard deviation
        eps_beta: float
            Decay rate
        eps_init: float
            Initial decay value
        eps_min: float
            Minimum decay value
        """
        # Base class initialization
        super().__init__(size, seed)

        self.mu = mu
        self.sigma = sigma
        self._eps_init = eps_init
        self._eps_min = eps_min
        self._eps_beta = eps_beta
        self.reset()

    def reset(self):
        """
        Reset the internal decay status to starting values
        """
        # Reset Epsilon
        self._eps = self._eps_init
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
        self._eps = max([np.exp(-self._eps_beta * self._eps_step), self._eps_min])
        self._eps_step += 1

        # Sample
        return np.random.normal(loc=self.mu,
                                scale=self.sigma,
                                size=self.size) * self._eps


class OUNoise(BaseNoise):
    """
    Ornstein-Uhlenbeck process also called Damped Random Walk (DRW).
    """
    def __init__(self, size, seed, mu=0., theta=.15, sigma=.2):
        """
        Initialize parameters and noise process

        Parameters
        ----------
        mu: float
            Initial value
        theta: float
            Theta variable, \theta > 0
        sigma: float
            Sigma variable, \sigma > 0
        """
        # Base class initialization
        super().__init__(size, seed)

        self.mu = mu * np.ones(self.size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """
        Reset the internal state to mean (mu)
        """
        self.state = copy(self.mu)

    def sample(self):
        """
        Update internal state and return it as a noise sample

        Return
        ------
        state: float
            Sample from the process
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
