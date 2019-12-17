"""
Handle configurations files
"""
import logging
from copy import deepcopy
from configobj import ConfigObj
from validate import Validator


class InvalidKey(ValueError):
    """
    Exception classes for no active section
    """

    def __init__(self, *args, **kwargs):
        msg = "Keys should have valid Python names"
        super().__init__(self, msg, *args, **kwargs)


class NoActiveSectionException(AttributeError):
    """
    Exception classes for no active section
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class ValidationError(ValueError):
    """
    Exception classes for invalid configuration
    """

    def __init__(self, validation_dict):
        message = "The following keys have invalid values:\n"
        message = message + self._failed_validation(validation_dict)
        super().__init__(message)

    def _failed_validation(self, _dict, prev_msg="", depth=0):
        msg = prev_msg
        for k, v in _dict.items():
            if v is True:
                continue
            elif isinstance(v, dict):
                sec = "\t" * depth + f"From section '{k}':\n"
                msg += self._failed_validation(v, sec, depth+1)
            elif v is False:
                msg += "\t" * depth + f"Mandatory key '{k}' is missing\n"
            else:
                msg += "\t" * depth + f"Key {k} raises {v}\n"
        return msg


# Main configuration class
class BisTrainConfiguration(ConfigObj):
    """
    Extended class with built-in validation
    """

    def __init__(self, *args, configspec="config.spec",
                 default_key="GLOBAL", **kwargs):
        super().__init__(*args, configspec=configspec, **kwargs)

        # Perform validation
        self.validator = Validator()
        self.validation = self.validate(self.validator, preserve_errors=True)
        if not isinstance(self.validation, dict):
            logging.info("File validation successfully!")
        else:
            logging.error("File validation failed!")
            raise ValidationError(self.validation)

        self._default_key = default_key

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            # Search on default key
            return self[self._default_key].__getitem__(key)

    def dict_copy(self):
        _copy = deepcopy(self)
        del _copy[self._default_key]
        # Add default values
        for k, v in self[self._default_key].items():
            _copy[k] = v

        return _copy


class LocalConfig():
    """
    Local configuration class with attribute accessors.
    """

    def __init__(self, section):
        try:
            # If section
            self._dict = section.dict()
            # Navigate upwards to add defaults
            while section.parent is not section:
                section = section.parent
            for k, v in section[section._default_key].items():
                self._dict[k] = v
        except AttributeError:
            self._dict = section

        print(self)
        print(self.__dir__())
        # Populates key values as attibutes
        for k, v in self._dict.items():
            if isinstance(v, dict):
                v = LocalConfig(v)
            if self._valid_key(k):
                print("aaa")
                self.__setattr__(k, v)
            else:
                raise InvalidKey()
        print(self.__dir__())

    def _valid_key(self, key):
        return " " not in key

    def __repr__(self):
        return repr(self._dict)

    def __str__(self):
        return str(self._dict)
