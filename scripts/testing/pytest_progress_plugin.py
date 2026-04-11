# Copyright (c) 2022 - 2026 Ali Hassani.
#
# Pytest plugin that writes live progress to a JSON file.
# Loaded via: pytest -p pytest_progress_plugin
#
# Writes to NATTEN_PROGRESS_FILE env var (if set), updating after each test.
# Format: {"collected": N, "passed": N, "failed": N, "skipped": N, "error": N}

import json
import os


def _progress_path():
    return os.environ.get("NATTEN_PROGRESS_FILE")


def _write(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


class ProgressPlugin:
    def __init__(self, path):
        self.path = path
        self.data = {
            "collected": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "xfailed": 0,
            "error": 0,
        }

    def pytest_collection_modifyitems(self, items):
        self.data["collected"] = len(items)
        _write(self.data, self.path)

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if hasattr(report, "wasxfail"):
                self.data["xfailed"] += 1
            elif report.passed:
                self.data["passed"] += 1
            elif report.failed:
                self.data["failed"] += 1
            elif report.skipped:
                self.data["skipped"] += 1
        elif report.when == "setup":
            if report.failed:
                self.data["error"] += 1
            elif report.skipped:
                if hasattr(report, "wasxfail"):
                    self.data["xfailed"] += 1
                else:
                    self.data["skipped"] += 1
        _write(self.data, self.path)

    def pytest_collectreport(self, report):
        if report.failed:
            self.data["error"] += 1
            _write(self.data, self.path)


def pytest_configure(config):
    path = _progress_path()
    if path:
        config.pluginmanager.register(ProgressPlugin(path), "natten_progress")
