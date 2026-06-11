import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_level: str = "INFO", log_dir: str = "logs"):
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=log_level,
    )
    logger.add(
        log_dir / "system.log",
        rotation="10 MB",
        retention="7 days",
        level=log_level,
    )

    return logger
