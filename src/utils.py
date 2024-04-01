import logging
from config.config import Config


def setup_logging():
    logging.basicConfig(
        filename=Config.get("log_path"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
