# Standard library imports
import logging

# Local application imports
from config.config import Config


def setup_logging():
    """
    Set up logging configuration using the log file path and format from the configuration.
    """
    logging.basicConfig(
        filename=Config.get("log_path"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
