"""
Handle configurations files
"""
from configparser import ConfigParser
from .mandatory import MANDATORY


class Configuration(dict):
    """
    Dictionary extended class with dot access of its values
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_configuration(file="config.yaml", replace=True):
    """
    Load the default configuration file and verify its values

    Parameters
    ----------
    file: str
        Path to configuration file
    replace: bool
        Weather or not replace missing values
    """
    # Load file
    config = ConfigParser()
    config.read(file)

    # Check values
    verify_values(config)

    # Save checked values
    config.write(file)

    return Configuration(config)


def verify_values(parameters, replace=False, mandatory=MANDATORY):
    """
    Verify if the configuration dictionary has all the needed keys

    Parameters
    ----------
    parameters: dict
        Parameters loaded from configuration file
    replace: bool
        Weather or not replace missing values with defaults
    mandatory: dict
        Dictionary with of mandatory "NUMERIC" and "CATEGORICAL" parameters
    """
    if replace:
        # Load default config
        config = ConfigParser().read('defaults.yaml')

    # Numeric values
    for key in mandatory['NUMERIC'].keys():
        # Check presence
        if key not in parameters.keys():
            msg = f"The configuration parameters {key} is missing."
            if replace:
                print("Some parameteres were missing, filling with default values..")
                parameters[key] = config[key]
            else:
                raise MissingParameterException(msg)
        # Check values
        else:
            _min = mandatory["NUMERIC"][key]["MIN"]
            _max = mandatory["NUMERIC"][key]["MAX"]
            if not(_min <= parameters[key] <= _max):
                msg = f"The configuration parameters {key} is invalid."
                raise InvalidParameterException(msg)


class MissingParameterException(IndexError):
    """
    Exception classes for missing mandatory configuration
    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class InvalidParameterException(ValueError):
    """
    Exception classes for invalid mandatory configuration
    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
