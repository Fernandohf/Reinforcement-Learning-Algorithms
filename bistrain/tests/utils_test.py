import os

import numpy as np
import pytest

from ..utils.configuration import (BisTrainConfiguration,
                                   NoActiveSectionException,
                                   ValidationError,
                                   MissingOptionError)
from ..utils.noise import GaussianNoise, OUNoise

LOCAL_FOLDER = os.path.dirname(__file__)
INVALID_FILE_1 = os.path.join(LOCAL_FOLDER, 'test_invalid_config_1.yaml')
INVALID_FILE_2 = os.path.join(LOCAL_FOLDER, 'test_invalid_config_2.yaml')
VALID_FILE = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
CONFIG_SPEC = os.path.join('config.spec')


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
        a.activate_sections("A2C")
        with pytest.raises(MissingOptionError):
            a.abc

    def test_activation(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        with pytest.raises(NoActiveSectionException):
            a.critic_hidden_size

    def test_dict_access1(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        assert a["GLOBAL"]["ACTION_SIZE"] == 2

    def test_dict_access2(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        assert a["GLOBAL"]["SEED"] == 42

    def test_dict_attr_access(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a.activate_sections("A2C")
        assert a.CRITIC["HIDDEN_SIZE"] == [256, 128]

    def test_attribute_access1(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a.activate_sections("GLOBAL")
        assert a.ACTION_SIZE == 2

    def test_attribute_access2(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a.activate_sections(["A2C", "TRAINING"])
        assert a.DEVICE == 'cpu'

    def test_attribute_access3(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a.activate_sections(["A2C", "ACTOR"])
        assert a.STATE_SIZE == 24

    def test_attribute_access4(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a.activate_sections(["A2C", "ACTOR"])
        assert a.HIDDEN_SIZE == [256, 32]

    def test_attribute_access5(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a.activate_sections("A2C")
        a.activate_subsection("CRITIC")
        assert a.HIDDEN_SIZE == [256, 128]

    def test_attribute_access6(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a.activate_sections("A2C")
        a.activate_subsection("CRITIC")
        assert a.STATE_SIZE == 24

    def test_attribute_access7(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a.activate_sections("A2C")
        a.activate_subsection("CRITIC")
        a.deactivate_subsection()
        with pytest.raises(MissingOptionError):
            a.HIDDEN_SIZE

    def test_default_key(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC,
                                  default_key="A2C")
        a.activate_sections("GLOBAL")
        assert isinstance(a.ACTOR, dict)


class TestGaussianNoise():
    """
    Test class to Gaussian Noise
    """
    @staticmethod
    def _create_noise():
        c = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        c["GLOBAL"]["ACTION_SIZE"] = 1
        c.activate_sections("EXPLORATION")
        n = GaussianNoise(c)
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
        c.activate_sections("EXPLORATION")
        n = OUNoise(c)
        return n

    def test_reset(self):
        n = self._create_noise()
        _ = [n.sample() for i in range(100)]
        n.reset()
        assert n.state == n.config.MEAN
