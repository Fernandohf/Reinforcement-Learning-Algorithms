"""
Tests for each implemeted agents
"""
import os

from torch.optim import Adam

from ..agents import A2CAgent
from ..utils.noise import GaussianNoise
from ..utils.configuration import BisTrainConfiguration
from ..networks.actors import FCActorContinuous


LOCAL_FOLDER = os.path.dirname(__file__)
VALID_FILE = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
VALID_FILE_A2C = os.path.join(LOCAL_FOLDER, 'test_valid_a2c.yaml')
CONFIG_SPEC = os.path.join('config.spec')


# TODO
class TestA2CAgent():
    """
    Test class to A2C Agent
    """

    def _create_agent(self, file=VALID_FILE):
        c = BisTrainConfiguration(file, configspec=CONFIG_SPEC)
        n = GaussianNoise(c["EXPLORATION"])
        a = A2CAgent(c["A2C"], n)
        return a

    def test_optimizer(self):
        a = self._create_agent()
        assert isinstance(a.actor.optimizer, Adam)

    def test_noise(self):
        a = self._create_agent()
        assert isinstance(a.noise, GaussianNoise)

    def test_policy(self):
        a = self._create_agent(VALID_FILE_A2C)
        assert isinstance(a.actor, FCActorContinuous)
