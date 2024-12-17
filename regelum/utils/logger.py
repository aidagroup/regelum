from loguru import logger
import sys


def setup_logger() -> None:
    """Configure the default logger for the package."""
    logger.remove()  # Remove default handler
    logger.add(
        sink=sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | - <level>{message}</level>",
        level="INFO",
    )


# Call setup on import
setup_logger()

# Export logger for use in other modules
__all__ = ["logger"]
