import logging
import os
import sys
import tempfile

import pytest
from natten.utils.log import _get_log_pipe, NattenLogger


@pytest.fixture(autouse=True)
def clean_env():
    old_pipe = os.environ.pop("NATTEN_LOG_PIPE", None)
    old_level = os.environ.get("NATTEN_LOG_LEVEL", None)
    os.environ["NATTEN_LOG_LEVEL"] = "debug"
    yield
    if old_pipe is not None:
        os.environ["NATTEN_LOG_PIPE"] = old_pipe
    else:
        os.environ.pop("NATTEN_LOG_PIPE", None)
    if old_level is not None:
        os.environ["NATTEN_LOG_LEVEL"] = old_level
    else:
        os.environ.pop("NATTEN_LOG_LEVEL", None)


def _make_logger(name: str) -> NattenLogger:
    # Use unique names to avoid shared logging.Logger instances
    logger = NattenLogger(name)
    logger.logger.propagate = False
    return logger


class TestGetLogPipe:
    def test_stdout_explicit(self):
        os.environ["NATTEN_LOG_PIPE"] = "stdout"
        assert _get_log_pipe() is sys.stdout

    def test_stdout_default(self):
        # Env var unset; should default to stdout
        os.environ.pop("NATTEN_LOG_PIPE", None)
        assert _get_log_pipe() is sys.stdout

    def test_stderr(self):
        os.environ["NATTEN_LOG_PIPE"] = "stderr"
        assert _get_log_pipe() is sys.stderr

    def test_dev_null(self):
        os.environ["NATTEN_LOG_PIPE"] = "/dev/null"
        assert _get_log_pipe() is None

    def test_valid_file(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            os.environ["NATTEN_LOG_PIPE"] = path
            assert _get_log_pipe() == path
        finally:
            os.unlink(path)

    def test_new_file_in_writable_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "natten.log")
            os.environ["NATTEN_LOG_PIPE"] = path
            result = _get_log_pipe()
            assert result == path
            # Touch should have created the file
            assert os.path.isfile(path)
            os.unlink(path)

    def test_invalid_path(self):
        os.environ["NATTEN_LOG_PIPE"] = "/nonexistent_dir/natten.log"
        assert _get_log_pipe() is None


def _read_log_file(path: str) -> str:
    with open(path) as f:
        return f.read()


class TestLogPipe:
    def test_stdout_logger(self):
        os.environ["NATTEN_LOG_PIPE"] = "stdout"
        logger = _make_logger("test.stdout")
        assert isinstance(logger.handler, logging.StreamHandler)
        assert logger.handler.stream is sys.stdout

    def test_stderr_logger(self):
        os.environ["NATTEN_LOG_PIPE"] = "stderr"
        logger = _make_logger("test.stderr")
        assert isinstance(logger.handler, logging.StreamHandler)
        assert logger.handler.stream is sys.stderr

    def test_dev_null_logger(self):
        os.environ["NATTEN_LOG_PIPE"] = "/dev/null"
        logger = _make_logger("test.devnull")
        assert isinstance(logger.handler, logging.NullHandler)

    def test_file_logger(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            os.environ["NATTEN_LOG_PIPE"] = path
            logger = _make_logger("test.file")
            assert isinstance(logger.handler, logging.FileHandler)
            logger.debug("hello from test")
            logger.handler.close()
            assert "hello from test" in _read_log_file(path)
        finally:
            os.unlink(path)

    def test_invalid_path_logger(self):
        os.environ["NATTEN_LOG_PIPE"] = "/nonexistent_dir/natten.log"
        logger = _make_logger("test.invalid")
        assert isinstance(logger.handler, logging.NullHandler)


class TestHandlerTypeSwitch:
    def test_stdout_then_stderr(self):
        os.environ["NATTEN_LOG_PIPE"] = "stdout"
        logger1 = _make_logger("test.type.1a")
        assert logger1.handler.stream is sys.stdout

        os.environ["NATTEN_LOG_PIPE"] = "stderr"
        logger2 = _make_logger("test.type.1b")
        assert logger2.handler.stream is sys.stderr

    def test_stdout_then_dev_null(self):
        os.environ["NATTEN_LOG_PIPE"] = "stdout"
        logger1 = _make_logger("test.type.2a")
        assert isinstance(logger1.handler, logging.StreamHandler)

        os.environ["NATTEN_LOG_PIPE"] = "/dev/null"
        logger2 = _make_logger("test.type.2b")
        assert isinstance(logger2.handler, logging.NullHandler)

    def test_dev_null_then_file(self):
        os.environ["NATTEN_LOG_PIPE"] = "/dev/null"
        logger1 = _make_logger("test.type.3a")
        assert isinstance(logger1.handler, logging.NullHandler)

        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            os.environ["NATTEN_LOG_PIPE"] = path
            logger2 = _make_logger("test.type.3b")
            assert isinstance(logger2.handler, logging.FileHandler)
            logger2.handler.close()
        finally:
            os.unlink(path)

    def test_file_then_stdout(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            os.environ["NATTEN_LOG_PIPE"] = path
            logger1 = _make_logger("test.type.4a")
            assert isinstance(logger1.handler, logging.FileHandler)
            logger1.handler.close()
        finally:
            os.unlink(path)

        os.environ["NATTEN_LOG_PIPE"] = "stdout"
        logger2 = _make_logger("test.type.4b")
        assert isinstance(logger2.handler, logging.StreamHandler)
        assert logger2.handler.stream is sys.stdout


class TestLogLevel:
    """Verify that messages below the configured log level are filtered out."""

    def _file_logger(self, name: str, path: str, level: str) -> NattenLogger:
        os.environ["NATTEN_LOG_LEVEL"] = level
        os.environ["NATTEN_LOG_PIPE"] = path
        return _make_logger(name)

    def test_debug_level_captures_all(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            logger = self._file_logger("test.level.debug", path, "debug")
            logger.debug("dbg msg")
            logger.info("info msg")
            logger.warning("warn msg")
            logger.handler.close()
            contents = _read_log_file(path)
            assert "dbg msg" in contents
            assert "info msg" in contents
            assert "warn msg" in contents
        finally:
            os.unlink(path)

    def test_info_level_filters_debug(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            logger = self._file_logger("test.level.info", path, "info")
            logger.debug("dbg msg")
            logger.info("info msg")
            logger.warning("warn msg")
            logger.handler.close()
            contents = _read_log_file(path)
            assert "dbg msg" not in contents
            assert "info msg" in contents
            assert "warn msg" in contents
        finally:
            os.unlink(path)

    def test_warning_level_filters_debug_and_info(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            logger = self._file_logger("test.level.warning", path, "warning")
            logger.debug("dbg msg")
            logger.info("info msg")
            logger.warning("warn msg")
            logger.handler.close()
            contents = _read_log_file(path)
            assert "dbg msg" not in contents
            assert "info msg" not in contents
            assert "warn msg" in contents
        finally:
            os.unlink(path)
