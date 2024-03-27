import logging
import logging.config


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        'my.packg': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        '__main__': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
    }
}


def get_logger(name, root_logger_level=''):
    if root_logger_level:
        LOGGING_CONFIG['loggers']['']['level'] = root_logger_level
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(name)
    return logger
