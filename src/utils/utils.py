# Standard library imports
import logging

# Local application imports
from config.config import get_config


def setup_logging():
    """
    Set up logging configuration using the log file path and format from the configuration.
    """
    logging.basicConfig(
        filename=get_config().LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
