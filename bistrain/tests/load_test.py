import os
import pytest
from ..load.configuration import BisTrainConfiguration, MissingParameterException, InvalidParameterException


class TestBisTrainConfiguration():
    """
    Basic test for BisTrainConfiguration class
    """
    def test_invalid1(self):
        file = os.path.join('tests', 'test_invalid_config.yaml')
        with pytest.raises(InvalidParameterException):
            BisTrainConfiguration(file)

    def test_invalid2(self):
        file = os.path.join('tests', 'test_invalid_config.yaml')
        with pytest.raises(KeyError):
            BisTrainConfiguration(file)

    def test_valid(self):
        file = os.path.join('tests', 'test_valid_config.yaml')
        a = BisTrainConfiguration(file)
        assert a.getvalue("AGENT", "action_size") == 2

    def test_missing(self):
        file = os.path.join('tests', 'test_missing_config.yaml')
        with pytest.raises(MissingParameterException):
            BisTrainConfiguration(file)
