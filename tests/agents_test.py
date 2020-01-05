"""
Tests for each implemented agents
"""
from torch.optim import Adam

from bistrain.agents import A2CAgent, DDPGAgent
from bistrain.noise import GaussianNoise, OUNoise
from bistrain.config.configuration import BisTrainConfiguration
from bistrain.networks.actors import FCActorContinuous, FCActorDiscrete
from . import CONFIG_A2C, CONFIG_DDPG, VALID_FILE, VALID_FILE_A2C, VALID_FILE_DDPG


# TODO - learning tests
class TestA2CAgent():
    """
    Test class to A2C Agent
    """

    def _create_agent(self, file=VALID_FILE):
        c = BisTrainConfiguration(file, configspec=CONFIG_A2C)
        env = lambda x: x
        env.config = c.get_localconfig("ENVIRONMENT")
        n = GaussianNoise(c.get_localconfig("EXPLORATION"))
        a = A2CAgent(c.get_localconfig("A2C"), n, env)
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


class TestDDPGAgent():
    """
    Test class to DDPG Agent
    """
    def _create_agent(self, file=VALID_FILE_DDPG):
        c = BisTrainConfiguration(file, configspec=CONFIG_DDPG)
        env = lambda x: x
        env.config = c.get_localconfig("ENVIRONMENT")
        n = OUNoise(c.get_localconfig("EXPLORATION"))
        a = DDPGAgent(c.get_localconfig("DDPG"), n, env)
        return a

    def test_optimizer(self):
        a = self._create_agent()
        assert isinstance(a.actor_local.optimizer, Adam)

    def test_noise(self):
        a = self._create_agent()
        assert isinstance(a.noise, OUNoise)

    def test_policy(self):
        a = self._create_agent()
        assert isinstance(a.actor_target, FCActorDiscrete)
