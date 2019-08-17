import logging
import sys

INFO_LOGGER = 'info_logger'
LOGGING_FORMAT = "%(asctime)s - %(message)s"

levels = {
    'error': logging.ERROR,
    'info': logging.INFO,
    'debug': logging.DEBUG
}


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(object, metaclass=Singleton):

    _logger = None

    def __init__(self):
        self._logger = logging.getLogger(INFO_LOGGER)

    def set_log_level(self, log_level):
        self._logger.setLevel(levels[log_level])

    def add_stream_handler(self, log_level):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(levels[log_level])
        stream_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
        self._logger.addHandler(stream_handler)

    def add_file_handler(self, log_level, file):
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(levels[log_level])
        file_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
        self._logger.addHandler(file_handler)


logger = Logger.__call__()._logger
