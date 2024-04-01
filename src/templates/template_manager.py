import json


class TemplateManager:
    templates = None

    @classmethod
    def load_templates(cls):
        if cls.templates is None:
            with open("src/chatbot/templates/template.json", "r") as f:
                cls.templates = json.load(f)

    @classmethod
    def get(cls, key, default="", **kwargs):
        if cls.templates is None:
            cls.load_templates()
        template = cls.templates.get(key, default)
        return template.format(**kwargs)
