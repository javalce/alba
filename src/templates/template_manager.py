import json
from config.config import Config


class TemplateManager:
    templates = None

    @classmethod
    def load_templates(cls):
        if cls.templates is None:
            templates_path = Config.get("templates_path")
            with open(templates_path, "r") as f:
                cls.templates = json.load(f)

    @classmethod
    def get(cls, key, default="", **kwargs):
        if cls.templates is None:
            cls.load_templates()
        template = cls.templates.get(key, default)
        return template.format(**kwargs)
