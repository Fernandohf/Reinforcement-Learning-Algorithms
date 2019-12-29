import os
from ..trainer import Trainer
from ..utils.configuration import BisTrainConfiguration


LOCAL_FOLDER = os.path.dirname(__file__)
VALID_FILE = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
VALID_FILE_A2C = os.path.join(LOCAL_FOLDER, 'test_valid_a2c.yaml')
VALID_FILE_DDPG = os.path.join(LOCAL_FOLDER, 'test_valid_ddpg.yaml')
CONFIG_SPEC = os.path.join('config.spec')


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

    def test_init(self):
        c = BisTrainConfiguration(VALID_FILE_A2C, configspec=CONFIG_SPEC)
        t = Trainer(c)
        assert t.config.EPISODES == 1000
