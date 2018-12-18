import logging

# This module contains method to create loggers

LOG_FILENAME = 'application.log'


def get_logger(name, log_level=logging.INFO):
    # create Logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create file handler and set level to info
    fh = logging.FileHandler(LOG_FILENAME)

    # create formatter
    formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')

    # add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add ch to Logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

