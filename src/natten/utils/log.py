#################################################################################################
# Copyright (c) 2022 - 2026 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################

import enum
import logging
import os
import sys

from natten.utils.environment import is_torch_compiling, parse_env_str

log_format = "| %(asctime)s | [[ %(name)s ]] [ %(levelname)s ]: %(message)s"


class LogLevel(enum.Enum):
    Default = 0
    Debug = 1
    Info = 2
    Warnings = 3
    Errors = 4
    Critical = 5


def _get_log_level() -> LogLevel:
    log_level = parse_env_str("NATTEN_LOG_LEVEL", "").lower()

    if log_level == "debug":
        return LogLevel.Debug
    elif log_level == "info":
        return LogLevel.Info
    elif log_level == "warning":
        return LogLevel.Warnings
    elif log_level == "error":
        return LogLevel.Errors
    elif log_level == "critical":
        return LogLevel.Critical

    return LogLevel.Default


_map_log_level = {
    LogLevel.Default: logging.INFO,
    LogLevel.Debug: logging.DEBUG,
    LogLevel.Info: logging.INFO,
    LogLevel.Warnings: logging.WARNING,
    LogLevel.Errors: logging.ERROR,
    LogLevel.Critical: logging.CRITICAL,
}


# Tests will stream into stderr instead of stdout
# It can be set to either stderr, stdout or any writeable file.
# Otherwise logging will be disabled.
def _get_log_pipe():
    log_pipe = parse_env_str("NATTEN_LOG_PIPE", "stdout")

    # Skip checking /dev/null writablity
    if log_pipe == "/dev/null":
        return None

    if log_pipe.lower() == "stderr":
        return sys.stderr

    if log_pipe.lower() == "stdout":
        return sys.stdout

    # Treat as file path; validate writability
    if os.path.isfile(log_pipe) and os.access(log_pipe, os.W_OK):
        return log_pipe

    try:
        open(log_pipe, "a").close()
        return log_pipe
    except OSError:
        pass

    return None


class NattenLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.log_level = _map_log_level[_get_log_level()]
        self.logger.setLevel(self.log_level)
        self.formatter = logging.Formatter(log_format)
        log_pipe = _get_log_pipe()
        if log_pipe in [sys.stderr, sys.stdout]:
            self.handler = logging.StreamHandler(log_pipe)
        elif isinstance(log_pipe, str):
            self.handler = logging.FileHandler(log_pipe)
        else:
            # Invalid / null
            self.handler = logging.NullHandler()  # type: ignore[assignment]
        self.handler.setLevel(self.log_level)
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

    def is_safe_to_log(self) -> bool:
        return not is_torch_compiling()

    def info(self, *args, **kwargs):
        if self.is_safe_to_log():
            self.logger.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        if self.is_safe_to_log():
            self.logger.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        if self.is_safe_to_log():
            self.logger.warning(*args, **kwargs)


def get_logger(name) -> NattenLogger:
    return NattenLogger(name)
