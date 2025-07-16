import logging

logger = logging.getLogger(__name__)

class SimpleTokenLogger():
    """Logs token usage to the standard logging system."""
    def log(self, prompt: int, completion: int) -> None:
        logger.info("Token usage - prompt: %d, completion: %d", prompt, completion)

