"""
Handle configurations files
"""
from configparser import ConfigParser
from mandatory import MANDATORY


# Exception classes
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


# Configuration class
class BisTrainConfiguration(ConfigParser):
    """
    Extended class with dot access and consistency check
    """
    BOOL_POS_VALUES = ["true", "ok", "yes", "on", "fine"]
    BOOL_NEG_VALUES = ["false", "no", "missing", "off", "not"]

    def __init__(self, file, mandatory=MANDATORY, **kwargs):
        super().__init__(**kwargs)

        # Load file
        _ = self.read(file)
        self._verify_values(mandatory)

    def getvalue(self, section, option):
        if section != self.default_section and not self.has_section(section):
            raise KeyError(section)
        try:
            # Try to return int
            return self.getint(section, option)
        except ValueError:
            try:
                # Try to return float
                return self.getfloat(section, option)
            except ValueError:
                # Try to return boolean
                str_value = self.get(section, option).lower()
                if str_value in BisTrainConfiguration.BOOL_POS_VALUES:
                    return True
                elif str_value in BisTrainConfiguration.BOOL_NEG_VALUES:
                    return False
                else:
                    # Otherwise return str
                    return str_value

    def _verify_values(self, mandatory):
        """
        Verify if the configuration file has all the needed keys

        Parameters
        ----------
        mandatory: dict
            Dictionary with of mandatory values
        """

        # Check keys and convert to numeric when possible
        for s in mandatory.keys():
            # Check presence
            man_options = set(mandatory[s].keys())
            curr_options = set(self[s].keys())
            if not man_options.issubset(curr_options):
                msg = f"The configuration parameters {man_options - curr_options} \
                        are missing on sections {s}."
                raise MissingParameterException(msg)
            # Check values
            else:
                for k in mandatory[s]:
                    if isinstance(mandatory[s][k], dict):
                        _min = mandatory[s][k]["MIN"]
                        _max = mandatory[s][k]["MAX"]
                        if not (_min <= self.getvalue(s, k) <= _max):
                            msg = f"The configuration parameters {k} is invalid."
                            raise InvalidParameterException(msg)
                    elif isinstance(mandatory[s][k], list):
                        if self.get(s, k) not in mandatory[s][k]:
                            msg = f"The configuration parameters {k} should be\
                                    one of these: {mandatory[s][k]}."
                            raise InvalidParameterException(msg)


if __name__ == "__main__":
    a = BisTrainConfiguration('config.yaml')
