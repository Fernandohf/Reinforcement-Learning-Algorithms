"""
"""
import os

import gym

from bistrain.trainer import Trainer
from bistrain.utils.configuration import BisTrainConfiguration
from bistrain.utils.noise import GaussianNoise
from bistrain.agents import A2CAgent


LOCAL_FOLDER = os.path.dirname(__file__)
VALID_FILE = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
VALID_FILE_A2C = os.path.join(LOCAL_FOLDER, 'test_valid_a2c.yaml')
VALID_FILE_DDPG = os.path.join(LOCAL_FOLDER, 'test_valid_ddpg.yaml')
CONFIG_SPEC = os.path.join('bistrain', 'config.spec')


class TestTrainer():
    """
    Test class for Trainer
    """
    def test_creation1(self):
        c = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        t = Trainer(c)
        assert t

    def test_creation2(self):
        c = BisTrainConfiguration(VALID_FILE_DDPG, configspec=CONFIG_SPEC)
        t = Trainer(c)
        assert t

    def test_creation3(self):
        c = BisTrainConfiguration(VALID_FILE_A2C, configspec=CONFIG_SPEC)
        t = Trainer(c)
        assert t

    def test_init1(self):
        c = BisTrainConfiguration(VALID_FILE_A2C, configspec=CONFIG_SPEC)
        t = Trainer(c)
        assert t.config.EPISODES == 1000

    def test_load_env(self):
        c = BisTrainConfiguration(VALID_FILE_A2C, configspec=CONFIG_SPEC)
        t = Trainer(c)
        assert isinstance(t.load_environment(), gym.Env)

    def test_load_agent(self):
        c = BisTrainConfiguration(VALID_FILE_A2C, configspec=CONFIG_SPEC)
        t = Trainer(c)
        assert isinstance(t.load_agent(), A2CAgent)

    def test_load_noise(self):
        c = BisTrainConfiguration(VALID_FILE_A2C, configspec=CONFIG_SPEC)
        t = Trainer(c)
        assert isinstance(t.load_noise(), GaussianNoise)

    def test_run(self):
        c = BisTrainConfiguration(VALID_FILE_A2C, configspec=CONFIG_SPEC)
        # Check if it runs
        c["TRAINER"]["EPISODES"] = 2
        c["TRAINER"]["N_ENVS"] = 2
        t = Trainer(c)
        t.run()
        assert isinstance(t.load_noise(), GaussianNoise)
