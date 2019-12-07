"""
Handle configurations files
"""
from configobj import ConfigObj
from validate import Validator


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

    def __init__(self, validation, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.msg = f"The following keys have invalid values:\n"
        for k, v in validation.items():
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
        if not isinstance(self.validation, dict):
            print("File validation successfully!")
        else:
            raise ValidationError(self.validation)

        self.active_sections = None

    def __getattr__(self, opt):
        """
        After calling 'activate sections' values can be retrieved as attrbutes,
        mainly to typing keys size when getting from same sections/subsections.

        Parameters
        ----------
        option: str
            Keyword of the 'active_sections'

        Return
        -------
        :
        """
        # if in keys
        if opt not in self.keys() and self.active_sections is None:
            raise NoActiveSectionException(
                "No active section has been defined yet.\
                Call 'activate_sections' before accessing values.")
        else:
            value = self
            for section in self.active_sections:
                value = value[section]
            return value[opt]

    def activate_sections(self, sections):
        """
        When a section is active all keys points to this section
        plus default options.
        """
        if not isinstance(sections, list):
            sections = [sections]
        self.active_sections = sections


if __name__ == "__main__":
    a = BisTrainConfiguration('config.yaml')
