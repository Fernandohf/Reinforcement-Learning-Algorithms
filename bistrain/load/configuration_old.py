"""
Handle configurations files
"""
from configparser import ConfigParser, NoOptionError
from .mandatory import MANDATORY


# Exception classes
class MissingParameterError(IndexError):
    """
    Exception classes for missing mandatory configuration
    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class NoActiveSectionException(Exception):
    """
    Exception classes for no active section
    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class InvalidParameterError(ValueError):
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
        self.active_section = None

    def __getattr__(self, opt):
        return self.getdefault(opt, None)

    def getdefault(self, option, default):
        if self.active_section is None:
            raise NoActiveSectionException("No active section was defined, \
                                            call methods 'activate_section(section)' first.")
        return self.getvalue(self.active_section, option, default)

    def getvalue(self, section, option, default=None):
        if section != self.default_section and not self.has_section(section):
            raise KeyError(section)
        try:
            # Try to return int
            return self.getint(section, option)
        # if no option return None
        except NoOptionError:
            return default
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
                    # Try to return list of int/float
                    list_str = str_value.split(",")
                    if (('[' == list_str[0][0] and ']' == list_str[-1][-1]) or
                       ('(' == list_str[0][0] and ')' == list_str[-1][-1])):
                        try:
                            return tuple([int(v.strip('[]() ')) for v in list_str if v.strip("[]() ") != ""])
                        except ValueError:
                            return tuple([float(v.strip('[]() ')) for v in list_str if v.strip("[]() ") != ""])
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
                raise MissingParameterError(msg)
            # Check values
            else:
                for k in mandatory[s]:
                    if isinstance(mandatory[s][k], dict):
                        _min = mandatory[s][k]["MIN"]
                        _max = mandatory[s][k]["MAX"]
                        if not (_min <= self.getvalue(s, k) <= _max):
                            msg = f"The configuration parameters {k} is invalid."
                            raise InvalidParameterError(msg)
                    elif isinstance(mandatory[s][k], list):
                        if self.get(s, k) not in mandatory[s][k]:
                            msg = f"The configuration parameters {k} should be\
                                    one of these: {mandatory[s][k]}."
                            raise InvalidParameterError(msg)

    def activate_section(self, section):
        """
        When a section is active all keys points to this section plus default options.
        """
        self.active_section = section


if __name__ == "__main__":
    a = BisTrainConfiguration('config.yaml')
