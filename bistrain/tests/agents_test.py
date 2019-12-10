"""
Tests for each implemeted agents
"""
import os

import torch
from torch.optim import Adam

from ..agents import A2CAgent
from ..utils.noise import GaussianNoise, OUNoise

LOCAL_FOLDER = os.path.dirname(__file__)
VALID_FILE = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')


# TODO
class TestA2CAgent():
    """
    Test class to A2C Agent
    """

    def _create_agent(self):
        a = A2CAgent(VALID_FILE)
        return a

    def test_optimizer(self):
        a = self._create_agent()
        assert isinstance(a.actor_optimizer, Adam)

    def test_noise(self):
        a = self._create_agent()
        assert isinstance(a.noise, GaussianNoise)
