import logging

def set_logger(name, filename):
    if name in logging.root.manager.loggerDict:
        logger =  logging.getLogger(name)
        return logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create a file handler with append mode
    file_handler = logging.FileHandler(filename, mode='w') 
    # Set the format for log messages
    formatter = logging.Formatter('%(levelname)s - %(asctime)s -- %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger