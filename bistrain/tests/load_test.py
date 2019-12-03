import os
import pytest
from ..load.configuration import BisTrainConfiguration, MissingParameterException, InvalidParameterException

LOCAL_FOLDER = os.path.dirname(__file__)


class TestBisTrainConfiguration():
    """
    Basic test for BisTrainConfiguration class
    """
    def test_invalid1(self):
        file = os.path.join(LOCAL_FOLDER, 'test_invalid_config_1.yaml')
        with pytest.raises(InvalidParameterException):
            BisTrainConfiguration(file)

    def test_invalid2(self):
        file = os.path.join(LOCAL_FOLDER, 'test_invalid_config_2.yaml')
        with pytest.raises(KeyError):
            BisTrainConfiguration(file)

    def test_valid(self):
        file = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
        a = BisTrainConfiguration(file)
        assert a.getvalue("AGENT", "action_size") == 2

    def test_missing(self):
        file = os.path.join(LOCAL_FOLDER, 'test_missing_config.yaml')
        with pytest.raises(MissingParameterException):
            BisTrainConfiguration(file)
