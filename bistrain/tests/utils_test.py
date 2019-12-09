import os

import numpy as np
import pytest

from ..utils.configuration import (BisTrainConfiguration,
                                   NoActiveSectionException, ValidationError)
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

    def test_invalid1(self):
        with pytest.raises(ValidationError):
            BisTrainConfiguration(INVALID_FILE_1, configspec=CONFIG_SPEC)

    def test_invalid2(self):
        with pytest.raises(ValidationError):
            BisTrainConfiguration(INVALID_FILE_2, configspec=CONFIG_SPEC)

    def test_valid1(self):
        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        assert a["AGENT"]["ACTION_SIZE"] == 2

    def test_valid2(self):

        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a.activate_sections("AGENT")
        assert a.ACTION_SIZE == 2

    def test_valid3(self):

        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a.activate_sections("TRAINING")
        assert a.DEVICE == 'cuda'

    def test_valid4(self):

        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a.activate_sections(["AGENT", "ACTOR"])
        assert a.HIDDEN_SIZE == [256, 32]

    def test_valid5(self):

        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a.activate_sections("AGENT")
        assert a.CRITIC["HIDDEN_SIZE"] == [256, 128]

    def test_valid6(self):

        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        a.activate_sections("AGENT")
        with pytest.raises(KeyError):
            assert a.abc

    def test_activation(self):

        a = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        with pytest.raises(NoActiveSectionException):
            a.critic_hidden_size


class TestGaussianNoise():
    """
    Test class to Gaussian Noise
    """
    @staticmethod
    def _create_noise():
        c = BisTrainConfiguration(VALID_FILE, configspec=CONFIG_SPEC)
        c["EXPLORATION"]["SIZE"] = 1
        c.activate_sections("EXPLORATION")
        n = GaussianNoise(c)
        return n

    def test_reset(self):
        n = self._create_noise()
        samples = [n.sample() for i in range(100)]
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
        c["EXPLORATION"]["SIZE"] = 1
        c["EXPLORATION"]["TYPE"] = 'ou'
        c.activate_sections("EXPLORATION")
        n = OUNoise(c)
        return n

    def test_reset(self):
        n = self._create_noise()
        samples = [n.sample() for i in range(100)]
        n.reset()
        assert n.state == n.config.MEAN


# TODO
class TestA2CAgent():
    """
    Test class to A2C Agent
    """

    def test_optimizer(self):
        pass

    def test_noise(self):
        pass
