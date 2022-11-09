import logging.config
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FORMATTER_VERCOSE = {
    "format": "{process} [{levelname}] ({asctime}) {pathname} <{lineno}> {name} : {message}",
    "style": "{",
}
FORMATTER_SIMPLE = {
    "format": "{process} {levelname} {asctime} <{lineno}> {name} : {message}",
    "style": "{",
}

HANDLER_CONSOLE = {
    "class": "logging.StreamHandler",
    "formatter": "simple",
}

LOGGERS = {
    "multipart": {"handlers": ["console"], "level": "WARNING"},
    "uvicorn": {"handlers": ["console"], "level": "WARNING"},
    "sqlalchemy.engine": {"handlers": ["console"], "level": "WARNING"},
    "sqlalchemy": {"handlers": ["console"], "level": "WARNING"},
    "opensearch": {"handlers": ["console"], "level": "WARNING"},
    "boto3": {"handlers": ["console"], "level": "WARNING"},
    "botocore": {"handlers": ["console"], "level": "WARNING"},
    "botocore.endpoint": {"handlers": ["console"], "level": "WARNING"},
    "urllib3": {"handlers": ["console"], "level": "WARNING"},
    "awswrangler": {"handlers": ["console"], "level": "INFO"},
    "requests_oauthlib": {"handlers": ["console"], "level": "WARNING"},
    "opentelemetry": {"handlers": ["console"], "level": "WARNING"},
    "pyathena.common": {"handlers": ["console"], "level": "WARNING"},
    "smart_open": {"handlers": ["console"], "level": "WARNING"},
    "smart_open.s3": {"handlers": ["console"], "level": "WARNING"},
}

LOGGING_CONFG = {
    "version": 1,
    "formatters": {
        "verbose": FORMATTER_VERCOSE,
        "simple": FORMATTER_SIMPLE,
    },
    "handlers": {
        "console": HANDLER_CONSOLE,
    },
    "loggers": LOGGERS,
    "root": {"handlers": ["console"], "level": os.getenv("ROOT_LOG_LEVEL", "DEBUG")},
}

logging.config.dictConfig(LOGGING_CONFG)