import logging
import logging.config
import os

def configure_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')

    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        
        'formatters': {
            'standard': {
                'format': '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                'datefmt': '%H:%M:%S'
            },
        },
        
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': 'DEBUG', 
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'logs/app.log',
                'formatter': 'standard',
                'level': 'INFO',
                'maxBytes': 5*1024*1024, # 5MB
                'backupCount': 3,
            },
        },
        
        'loggers': {
            '': { 
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': True
            },
                 
            'urllib3': {
                'level': 'WARNING'
            }
        }
    }
    
    logging.config.dictConfig(LOGGING_CONFIG)