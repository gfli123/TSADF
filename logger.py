import logging
import os
import sys


def load_logger(logger_name='VSR', log_level0=logging.DEBUG, log_level1=logging.INFO,
                log_level2=logging.WARNING, log_level3=logging.ERROR):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    # Creating a logger object
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level0)

    # Create a console processor and set the level
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(log_level0)
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    console.setFormatter(logging.Formatter(format_str))

    # Create the first file processor and set the level
    file_handler1 = logging.FileHandler('./results/all.txt', 'w')
    file_handler1.setLevel(log_level0)
    file_handler1.setFormatter(logging.Formatter(format_str))

    # Create the second file processor and set the level
    file_handler2 = logging.FileHandler('./results/test_detail.txt', 'w')
    file_handler2.setLevel(log_level1)
    file_handler2.setFormatter(logging.Formatter(format_str))

    # Create the third file processor and set the level
    file_handler3 = logging.FileHandler('./results/loss_and_test.txt', 'a')
    file_handler3.setLevel(log_level2)
    file_handler3.setFormatter(logging.Formatter(format_str))

    # Create the fourth file processor and set the level
    file_handler4 = logging.FileHandler('./results/final_results.txt', 'a')
    file_handler4.setLevel(log_level3)
    file_handler4.setFormatter(logging.Formatter(format_str))

    # Adding Processors to Logger
    logger.addHandler(console)
    logger.addHandler(file_handler1)
    logger.addHandler(file_handler2)
    logger.addHandler(file_handler3)
    logger.addHandler(file_handler4)

    return logger


