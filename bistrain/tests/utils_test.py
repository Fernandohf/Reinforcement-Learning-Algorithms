import os

import pytest

from ..utils.configuration import (BisTrainConfiguration,
                                   NoActiveSectionException, ValidationError)

LOCAL_FOLDER = os.path.dirname(__file__)
CONFIG_SPEC = os.path.join('config.spec')


class TestBisTrainConfiguration():
    """
    Tests for BisTrainConfiguration class
    """

    def test_invalid1(self):
        file = os.path.join(LOCAL_FOLDER, 'test_invalid_config_1.yaml')
        with pytest.raises(ValidationError):
            BisTrainConfiguration(file, configspec=CONFIG_SPEC)

    def test_invalid2(self):
        file = os.path.join(LOCAL_FOLDER, 'test_invalid_config_2.yaml')
        with pytest.raises(ValidationError):
            BisTrainConfiguration(file, configspec=CONFIG_SPEC)

    def test_valid1(self):
        file = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
        a = BisTrainConfiguration(file, configspec=CONFIG_SPEC)
        assert a["AGENT"]["ACTION_SIZE"] == 2

    def test_valid2(self):
        file = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
        a = BisTrainConfiguration(file, configspec=CONFIG_SPEC)
        a.activate_sections("AGENT")
        assert a.ACTION_SIZE == 2

    def test_valid3(self):
        file = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
        a = BisTrainConfiguration(file, configspec=CONFIG_SPEC)
        a.activate_sections("TRAINING")
        assert a.DEVICE == 'cuda'

    def test_valid4(self):
        file = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
        a = BisTrainConfiguration(file, configspec=CONFIG_SPEC)
        a.activate_sections(["AGENT", "ACTOR"])
        assert a.HIDDEN_SIZE == [256, 32]

    def test_valid5(self):
        file = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
        a = BisTrainConfiguration(file, configspec=CONFIG_SPEC)
        a.activate_sections("AGENT")
        assert a.CRITIC["HIDDEN_SIZE"] == [256, 128]

    def test_valid6(self):
        file = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
        a = BisTrainConfiguration(file, configspec=CONFIG_SPEC)
        a.activate_sections("AGENT")
        with pytest.raises(KeyError):
            assert a.abc

    def test_activation(self):
        file = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
        a = BisTrainConfiguration(file, configspec=CONFIG_SPEC)
        with pytest.raises(NoActiveSectionException):
            a.critic_hidden_size
