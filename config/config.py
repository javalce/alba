import json


class Config:
    """
    A class for managing configuration settings.

    This class loads the configuration from a JSON file and provides a method
    to retrieve configuration values by key.
    """

    config = None

    @classmethod
    def load_config(cls):
        """
        Load the configuration from the JSON file.

        This method reads the configuration from the 'config/config.json' file
        and stores it in the `config` class variable. If the configuration is
        already loaded, this method does nothing.
        """
        if cls.config is None:
            with open("config/config.json", "r") as f:
                cls.config = json.load(f)

    @classmethod
    def get(cls, key):
        """
        Get the value of a configuration setting by key.

        This method retrieves the value of the specified `key` from the loaded
        configuration. If the configuration is not yet loaded, it first calls
        the `load_config` method to load the configuration from the JSON file.

        Args:
            key (str): The key of the configuration setting to retrieve.

        Returns:
            The value of the configuration setting, or None if the key is not found.
        """
        if cls.config is None:
            cls.load_config()
        return cls.config.get(key)
