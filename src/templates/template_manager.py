import json
from config.config import get_config


class TemplateManager:
    """
    A class for managing templates loaded from a JSON file.
    """

    templates = None

    @classmethod
    def load_templates(cls):
        """
        Load templates from the JSON file specified in the configuration.
        This method is called automatically when accessing templates if they are not already loaded.
        """
        config = get_config()

        if cls.templates is None:
            templates_path = config.TEMPLATES_PATH
            with open(templates_path, "r") as f:
                cls.templates = json.load(f)

    @classmethod
    def get(cls, key: str, default: str = "", **kwargs) -> str:
        """
        Retrieve a template by its key and format it with the provided keyword arguments.

        Args:
            key (str): The key of the template to retrieve.
            default (str, optional): The default value to return if the key is not found. Defaults to an empty string.
            **kwargs: Keyword arguments to format the template.

        Returns:
            str: The formatted template.
        """
        if cls.templates is None:
            cls.load_templates()
        template = cls.templates.get(key, default)
        return template.format(**kwargs)
