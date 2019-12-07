"""
Handle configurations files
"""
from configobj import ConfigObj
from validate import Validator


class NoActiveSectionException(Exception):
    """
    Exception classes for no active section
    """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class ValidationError(ValueError):
    """
    Exception classes for invalid configuration
    """

    def __init__(self, validation, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.msg = f"The following keys have invalid values:\n"
        for k, v in validation:
            if not v:
                self.msg += f"Key {k} raises {v}\n"


# Configuration class
class BisTrainConfiguration(ConfigObj):
    """
    Extended class with dot access and built-in validation check
    """

    def __init__(self, *args, configspec="config.spec", **kwargs):
        super().__init__(*args, configspec=configspec, **kwargs)

        # Perform validation
        self.validator = Validator()
        self.validation = self.validate(self.validator, preserve_errors=True)
        if self.validation:
            print("File validation successfully!")
        else:
            raise ValidationError(self.validation)

        self.active_section = None

    def __getattr__(self, opt):
        # if in keys
        if opt not in self.keys() and self.active_section is None:
            raise AttributeError
        try:
            return self.__getitem__(self.active_section).__getitem__(opt)
        except KeyError:
            return self.__getitem__(opt)

    def activate_section(self, section):
        """
        When a section is active all keys points to this section
        plus default options.
        """
        self.active_section = section


if __name__ == "__main__":
    a = BisTrainConfiguration('config.yaml')
