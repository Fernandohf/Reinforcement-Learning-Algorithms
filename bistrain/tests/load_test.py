from ..load.configuration import Configuration, load_configuration, verify_values, MissingParameterException, InvalidParameterException


class TestBisTrainConfiguration():
    """
    Basic test for BisTrainConfiguration class
    """
    def test_feat_replace():
        a = load_configuration()
