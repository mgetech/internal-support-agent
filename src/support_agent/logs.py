"""Logging rules for the API. They are applied when the app starts, after Uvicorn has set
up its own logging.

Only the loggers under `support_agent` follow the level of the app (`LOG_LEVEL`). The root
logger and every other logger stay at WARNING, so a library cannot write its details to
the log, whatever level the app uses. This is a list of what is allowed, not a list of
libraries to silence: a library that is added later is quiet from the start. The one
exception is `uvicorn.error`, which stays at INFO so the server's start and stop lines are
shown. It has no request data. An error in a route never reaches it, because the API
answers it first (see `log_request` in api.py).

Uvicorn's access log is turned off. It has the client address and the query string, and a
query can have an employee id. The API writes its own access line instead: method, path
without the query, status and time.

A log line has a request id, a trace id, an event name, a time or an error type. Never
message text, tool arguments or results, an answer, an employee id or an exception message.
"""

from __future__ import annotations

import logging

APP_LOGGER = "support_agent"
# loggers that keep INFO. Uvicorn writes its start and stop lines here.
INFO_LOGGERS = ("uvicorn.error",)
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

access_logger = logging.getLogger(f"{APP_LOGGER}.access")


def configure_logging(level: str = "INFO") -> None:
    """Set the app's loggers to `level` and everything else to WARNING. Writes to stderr."""
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    # a library can set a level of its own on its logger, so every logger that exists is set
    for name, logger in logging.root.manager.loggerDict.items():
        if not isinstance(logger, logging.Logger):
            continue
        is_app_child = name.startswith(f"{APP_LOGGER}.")
        # a logger under support_agent has no level of its own, so it follows the app's
        logger.setLevel(logging.NOTSET if is_app_child else logging.WARNING)
    for name in INFO_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO)
    logging.getLogger(APP_LOGGER).setLevel(level.upper())
    logging.getLogger("uvicorn.access").disabled = True

    if not any(getattr(handler, "is_app_handler", False) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.is_app_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
