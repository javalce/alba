import json


class Config:
    config = None

    @classmethod
    def load_config(cls):
        if cls.config is None:
            with open("config/config.json", "r") as f:
                cls.config = json.load(f)

    @classmethod
    def get(cls, key):
        if cls.config is None:
            cls.load_config()
        return cls.config.get(key)
