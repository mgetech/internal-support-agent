"""The logging rules. The access log test starts the app on a real port with Uvicorn, because
only a real server writes Uvicorn's access line. The database must be up (make db-up).
"""

import json
import logging
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest
import uvicorn

from support_agent.api import app
from support_agent.logs import configure_logging

EMPLOYEE = "emp_003"
MARKER = "marker-5d2e8b17"
HR_PARTNER = "emp_005"
THIRD_PARTY = [
    "httpx",
    "httpx2",
    "httpcore",
    "httpcore.connection",
    "urllib3",
    "langfuse",
    "uvicorn",
    "some.other.library",
]


class Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def every_logger():
    loggers = logging.root.manager.loggerDict.values()
    return [logging.getLogger(), *(x for x in loggers if isinstance(x, logging.Logger))]


@pytest.fixture
def restore_logging():
    """Puts every logger back as it was, because configure_logging changes all of them."""
    saved = {
        logger: (logger.level, logger.disabled, list(logger.handlers), logger.propagate)
        for logger in every_logger()
    }
    yield
    for logger, (level, disabled, handlers, propagate) in saved.items():
        logger.setLevel(level)
        logger.disabled = disabled
        logger.handlers = handlers
        logger.propagate = propagate


def get_free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --- the allowlist ---


def test_with_the_app_at_debug_only_the_app_loggers_are_at_debug(restore_logging):
    # libraries as they are when a debug level was set for all of them
    logging.getLogger().setLevel(logging.DEBUG)
    for name in THIRD_PARTY:
        logging.getLogger(name).setLevel(logging.DEBUG)

    configure_logging("DEBUG")

    assert logging.getLogger().level == logging.WARNING
    for name in THIRD_PARTY:
        assert logging.getLogger(name).level == logging.WARNING, name
        assert not logging.getLogger(name).isEnabledFor(logging.INFO), name
    assert logging.getLogger("support_agent").level == logging.DEBUG
    assert logging.getLogger("support_agent.api").isEnabledFor(logging.DEBUG)
    assert logging.getLogger("support_agent.access").isEnabledFor(logging.DEBUG)


def test_uvicorn_error_keeps_info_so_the_server_lifecycle_lines_are_shown(restore_logging):
    logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)

    configure_logging("WARNING")

    assert logging.getLogger("uvicorn.error").level == logging.INFO
    # the other Uvicorn loggers stay quiet
    assert logging.getLogger("uvicorn").level == logging.WARNING


def test_a_library_logger_made_after_the_setup_is_quiet_too(restore_logging):
    configure_logging("DEBUG")

    assert not logging.getLogger("a.library.made.later").isEnabledFor(logging.INFO)


def test_the_app_follows_the_level_that_is_set(restore_logging):
    configure_logging("WARNING")

    assert not logging.getLogger("support_agent.api").isEnabledFor(logging.INFO)
    assert logging.getLogger("support_agent.api").isEnabledFor(logging.WARNING)


def test_a_debug_line_of_the_app_is_written_and_one_of_a_library_is_not(restore_logging):
    lines = Lines()
    configure_logging("DEBUG")
    logging.getLogger().addHandler(lines)

    logging.getLogger("support_agent.api").debug("from the app")
    logging.getLogger("httpcore").debug("from a library")
    logging.getLogger("httpx2").info("from another library")
    logging.getLogger("urllib3").warning("a warning from a library")

    assert lines.lines == ["from the app", "a warning from a library"]


def test_the_handler_is_added_once(restore_logging):
    configure_logging()
    configure_logging()

    handlers = logging.getLogger().handlers
    assert sum(getattr(handler, "is_app_handler", False) for handler in handlers) == 1


# --- a real server ---


@contextmanager
def running_app():
    """Runs the app with Uvicorn on a free port. Gives the port and the log lines of the
    loggers of Uvicorn and of the app.
    """
    port = get_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port)  # applies Uvicorn's logging
    lines = Lines()
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "support_agent"):
        logging.getLogger(name).addHandler(lines)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "the server did not start"
            time.sleep(0.05)
        yield port, lines.lines
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_the_access_line_has_no_employee_id_from_the_query_and_no_client_address(
    seeded_db, restore_logging
):
    with running_app() as (port, lines):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/decisions?subject_employee_id={EMPLOYEE}",
            headers={"X-Employee-Id": HR_PARTNER},
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 200

    # the start-up line names the address the server listens on. That is not a client.
    text = " | ".join(line for line in lines if "running on" not in line)
    assert EMPLOYEE not in text
    assert "127.0.0.1" not in text
    assert "localhost" not in text
    access = [line for line in lines if "/decisions" in line]
    assert len(access) == 1
    assert re.fullmatch(r"GET /decisions 200 \d+ms", access[0])


def test_the_server_lifecycle_lines_are_shown(restore_logging):
    with running_app() as (_, lines):
        pass

    assert any("Application startup complete" in line for line in lines)
    assert any("Finished server process" in line for line in lines)


@pytest.fixture
def boom_route():
    """A route that fails with an error whose message has the marker."""

    def boom():
        raise ValueError(f"cannot read the record of {EMPLOYEE}: {MARKER}")

    app.add_api_route("/boom", boom)
    yield
    app.router.routes.pop()


def test_an_error_in_a_route_is_a_500_with_the_request_id_and_no_log_line_has_its_message(
    boom_route, restore_logging
):
    with running_app() as (port, lines), pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/boom")

    body = json.loads(caught.value.read())
    assert caught.value.code == 500
    assert body == {"detail": "the request failed", "request_id": body["request_id"]}
    assert re.fullmatch(r"[0-9a-f]{32}", body["request_id"])
    text = " | ".join(lines)
    assert MARKER not in text
    assert EMPLOYEE not in text
    assert "Traceback" not in text
    assert f"request_failed error_type=ValueError request_id={body['request_id']}" in lines
    assert "POST /boom" not in text and "GET /boom 500" in text
