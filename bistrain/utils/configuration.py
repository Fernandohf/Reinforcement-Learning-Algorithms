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

    def __init__(self, *args, interpolation="template",
                 configspec="config.spec", **kwargs):
        super().__init__(*args, interpolation="template",
                         configspec=configspec, **kwargs)

        # Perform validation
        self.validator = Validator()
        self.validation = self.validate(self.validator, preserve_errors=True)
        print(self.validation)
        if not isinstance(self.validation, dict):
            print("File validation successfully!")
        else:
            raise ValidationError(self.validation)

        self._active_sections = None

    def __getattr__(self, opt):
        """
        After calling 'activate sections' values can be retrieved as attrbutes,
        mainly to typing keys size when getting from same sections/subsections.

        Parameters
        ----------
        option: str
            Key present on 'active_sections'

        """
        # if in keys
        if opt not in self.keys() and self._active_sections is None:
            raise NoActiveSectionException(
                "No active section has been defined yet.\
                Call 'activate_sections' before accessing values.")
        else:
            value = self
            for section in self._active_sections:
                value = value[section]
            return value[opt]

    @property
    def active_sections(self):
        return self._active_sections

    def activate_sections(self, sections):
        """
        When sections are active, values can be directly accessed with
        attribute.

        Parameters
        ----------
        sections: str or list
            Section to be used by default when accessing values as attributes
        """
        if not isinstance(sections, list):
            sections = [sections]
        self._active_sections = sections

    def activate_subsection(self, subsection):
        """
        Append subsection to active sections.

        Parameters
        ----------
        subsection: str
            Subsection to be activated
        """
        sections = self._active_sections + [subsection]
        self.activate_sections(sections)

    def deactivate_subsection(self):
        """
        Remove last active subsection
        """
        sections = self._active_sections[:-1]
        self.activate_sections(sections)


if __name__ == "__main__":
    a = BisTrainConfiguration('config.yaml')
