import logging
import os


def get_logger(name=__name__):
    logger = logging.getLogger(name)
    # 환경 변수에서 로깅 레벨 가져오기 (기본값: INFO)
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
