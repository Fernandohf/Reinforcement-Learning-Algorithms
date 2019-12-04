import os
import pytest
from ..load.configuration import BisTrainConfiguration, MissingParameterError, InvalidParameterError

LOCAL_FOLDER = os.path.dirname(__file__)


class TestBisTrainConfiguration():
    """
    Basic test for BisTrainConfiguration class
    """
    def test_invalid1(self):
        file = os.path.join(LOCAL_FOLDER, 'test_invalid_config_1.yaml')
        with pytest.raises(InvalidParameterError):
            BisTrainConfiguration(file)

    def test_invalid2(self):
        file = os.path.join(LOCAL_FOLDER, 'test_invalid_config_2.yaml')
        with pytest.raises(KeyError):
            BisTrainConfiguration(file)

    def test_valid1(self):
        file = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
        a = BisTrainConfiguration(file)
        assert a.getvalue("AGENT", "action_size") == 2

    def test_valid2(self):
        file = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
        a = BisTrainConfiguration(file)
        a.activate_section("AGENT")
        assert a.action_size == 2

    def test_valid3(self):
        file = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
        a = BisTrainConfiguration(file)
        a.activate_section("TRAINING")
        assert a.DEVICE == 'cuda'

    def test_missing(self):
        file = os.path.join(LOCAL_FOLDER, 'test_missing_config.yaml')
        with pytest.raises(MissingParameterError):
            BisTrainConfiguration(file)
