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


class MissingOptionError(KeyError):
    """
    Exception classes for missing mandatory configuration
    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


# Configuration class
class BisTrainConfiguration(ConfigObj):
    """
    Extended class with dot access and built-in validation check
    """

    def __init__(self, *args, configspec="config.spec",
                 default_key="DEFAULT", **kwargs):
        super().__init__(*args, configspec=configspec, **kwargs)

        # Perform validation
        self.validator = Validator()
        self.validation = self.validate(self.validator, preserve_errors=True)
        print(self.validation)
        if not isinstance(self.validation, dict):
            print("File validation successfully!")
        else:
            raise ValidationError(self.validation)

        self._active_sections = None
        self._default_key = default_key

    def __getattr__(self, opt):
        """
        After calling 'activate sections' values are retrieved as attributes,
        mainly to reduce typing large nested keys names after multiple
        sections/subsections. Prioritizes inner sections in the search.

        Parameters
        ----------
        opt: str
            Key present on 'active_sections'

        Return
        ------
        d[opt]: dict value
            Nested key with the given key
        """
        # if in keys
        if opt not in self.keys() and self._active_sections is None:
            raise NoActiveSectionException(
                "No active section has been defined yet.\
                Call 'activate_sections' before accessing values.")
        else:
            # Search final section
            _dicts = self._get_dict([self._default_key] +
                                    self._active_sections)
            for d in _dicts:
                try:
                    # Deepest section
                    return d[opt]
                except KeyError:
                    # Search upper sections
                    continue
            # Case key not found
            raise MissingOptionError(opt)

    def _get_dict(self, sections):
        """
        Auxiliar function to retrieve nested dict in access order
        """
        dicts = []
        value = self
        for section in self._active_sections:
            dicts.append(value[section])
            value = value[section]
        # Depth first
        return reversed(dicts)

    @property
    def active_sections(self):
        return self._active_sections

    def activate_sections(self, sections=None):
        """
        When sections are active, values can be directly accessed with
        attribute.

        Parameters
        ----------
        sections: str or list
            Section to be used by default when accessing values as attributes
        """
        if sections is None:
            self._active_sections = [self._default_key]
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
