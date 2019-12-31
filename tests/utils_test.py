import os

import numpy as np
import pytest

from bistrain.utils.configuration import (BisTrainConfiguration,
                                          LocalConfig,
                                          ValidationError,
                                          InvalidKey)
from bistrain.utils.noise import GaussianNoise, OUNoise

LOCAL_FOLDER = os.path.dirname(__file__)
INVALID_FILE_1 = os.path.join(LOCAL_FOLDER, 'test_invalid_config_1.yaml')
INVALID_FILE_2 = os.path.join(LOCAL_FOLDER, 'test_invalid_config_2.yaml')
VALID_FILE = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
CONFIG_SPEC = os.path.join('bistrain', 'config.spec')


class TestBisTrainConfiguration():
    """
    Tests for BisTrainConfiguration class
    """

    def test_validation1(self):
        with pytest.raises(ValidationError):
            BisTrainConfiguration(INVALID_FILE_1, configspec=CONFIG_SPEC)

    def test_validation2(self):
        with pytest.raises(ValidationError):
            BisTrainConfiguration(INVALID_FILE_2, configspec=CONFIG_SPEC)

    def test_invalid_key(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        with pytest.raises(KeyError):
            a["abc"]

    def test_dict_access1(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        assert a["GLOBAL"]["ACTION_SIZE"] == 1

    def test_dict_access2(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        assert a["GLOBAL"]["SEED"] == 42

    def test_dict_access3(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        assert a["A2C"]["CRITIC"]["HIDDEN_SIZE"] == [256, 128]

    def test_dict_access4(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        assert a["ACTION_SIZE"] == 1

    def test_dict_access5(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        assert a["A2C"]["TRAINING"]["GAMMA"] == 0.99

    def test_dict_access6(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        assert a["STATE_SIZE"] == 3

    def test_default_key(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC,
                                  default_key="EXPLORATION")
        assert a["TYPE"] == 'gaussian'

    def test_dict_copy(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        b = a.dict_copy()
        b["GLOBAL"] = 1
        assert a["GLOBAL"] != b["GLOBAL"]


class TestLocalConfig():
    """
    Testing local config class
    """
    def test_initiation(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        assert LocalConfig(a)

    def test_attr_access1(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        b = LocalConfig(a["EXPLORATION"])
        assert b.TYPE == 'gaussian'

    def test_attr_access2(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        b = LocalConfig(a["A2C"])
        assert b.ACTOR.LR == 0.001

    def test_attr_access3(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        b = LocalConfig(a["A2C"])
        assert b.SEED == 42

    def test_attr_access4(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        b = LocalConfig(a["A2C"])
        c = LocalConfig(b.CRITIC)
        assert c.OPTIMIZER == 'adam'

    def test_invalidkey1(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a["A2C"]["a b c"] = 4
        with pytest.raises(InvalidKey):
            LocalConfig(a["A2C"])

    def test_invalidkey2(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a["A2C"]["items"] = 4
        with pytest.raises(InvalidKey):
            LocalConfig(a["A2C"])


class TestGaussianNoise():
    """
    Test class to Gaussian Noise
    """
    @staticmethod
    def _create_noise():
        c = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        c["GLOBAL"]["ACTION_SIZE"] = 1
        n = GaussianNoise(c["EXPLORATION"])
        return n

    def test_reset(self):
        n = self._create_noise()
        _ = [n.sample() for i in range(100)]
        n.reset()
        assert n._eps_step == 0

    def test_decay(self):
        n = self._create_noise()
        samples = np.array([n.sample() for i in range(1000)])
        assert abs(samples.min()) >= n.config.EPS_MIN


class TestOUNoise():
    """
    Test class to OU Noise
    """
    @staticmethod
    def _create_noise():
        c = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        c["GLOBAL"]["ACTION_SIZE"] = 1
        c["EXPLORATION"]["TYPE"] = 'ou'
        n = OUNoise(c["EXPLORATION"])
        return n

    def test_reset(self):
        n = self._create_noise()
        _ = [n.sample() for i in range(100)]
        n.reset()
        assert n.state == n.config.MEAN
