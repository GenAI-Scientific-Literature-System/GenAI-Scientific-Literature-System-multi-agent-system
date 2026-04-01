import logging
import os

def get_logger(name: str) -> logging.Logger:
    os.makedirs("logs", exist_ok = True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s|%(message)s")
        file_handler = logging.FileHandler("logs/system.log", encoding="utf-8")
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger