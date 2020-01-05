"""
"""
import gym

from bistrain.trainer import Trainer
from bistrain.config.configuration import BisTrainConfiguration
from bistrain.noise import GaussianNoise
from bistrain.agents import A2CAgent
from . import CONFIG_SPEC, VALID_FILE, VALID_FILE_A2C, CONFIG_A2C


class TestTrainer():
    """
    Test class for Trainer
    """
    def test_creation1(self):
        c = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        t = Trainer(c)
        assert t

    def test_creation2(self):
        t = Trainer(VALID_FILE)
        assert t

    def test_creation3(self):
        c = BisTrainConfiguration(VALID_FILE_A2C, configspec=CONFIG_A2C)
        t = Trainer(c)
        assert t

    def test_init1(self):
        c = BisTrainConfiguration(VALID_FILE_A2C, configspec=CONFIG_A2C)
        t = Trainer(c)
        assert t.config.EPISODES == 1000

    def test_load_env(self):
        c = BisTrainConfiguration(VALID_FILE_A2C, configspec=CONFIG_A2C)
        t = Trainer(c)
        assert isinstance(t.load_environment(None), gym.Env)

    def test_load_agent(self):
        c = BisTrainConfiguration(VALID_FILE_A2C, configspec=CONFIG_A2C)
        t = Trainer(c)
        assert isinstance(t.load_agent(t.load_environment(None)), A2CAgent)

    def test_load_noise(self):
        c = BisTrainConfiguration(VALID_FILE_A2C, configspec=CONFIG_A2C)
        t = Trainer(c)
        assert isinstance(t.load_noise(), GaussianNoise)

    def test_run(self):
        c = BisTrainConfiguration(VALID_FILE_A2C, configspec=CONFIG_A2C)
        # Check if it runs
        c["TRAINER"]["EPISODES"] = 2
        c["TRAINER"]["N_ENVS"] = 2
        t = Trainer(c)
        t.run()
        assert isinstance(t.load_noise(), GaussianNoise)
