# Copyright (c) 2022 - 2026 Ali Hassani.
# Made largely with Claude Code
#
# Progress monitor for parallel test execution.
# Polls LOG_DIR/.status/ for per-test status files.
#
# Each file is named after the test and contains key=value lines:
#   start=<epoch>          written when test begins
#   end=<epoch>            written when test finishes
#   rc=<exit_code>         written when test finishes
#   gpu=<id>               GPU index (-1 = CPU)
#   worker=<id>            worker index
#
# States:
#   empty / no start  → not started (○)
#   start only        → running (●)
#   end XOR rc        → invalid (⚠)
#   end + rc          → done (✓ if rc=0, ✗ otherwise)
#
# Usage: monitor.py --log-dir DIR

import json
import os
import signal
import sys
import time


def _handle_term(*_):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)  # next TERM kills immediately
    sys.exit(0)


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"

REFRESH_INTERVAL = 1.0

# States
NOT_STARTED = "not_started"
RUNNING = "running"
INVALID = "invalid"
PASSED = "passed"
FAILED = "failed"


def format_elapsed(seconds):
    hrs, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def format_time(epoch):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


def read_status(path):
    try:
        with open(path) as f:
            return dict(line.strip().split("=", 1) for line in f if "=" in line)
    except (OSError, ValueError):
        return {}


def classify(fields):
    """Returns (state, start, end, rc, gpu, worker)."""
    raw_start = fields.get("start")
    raw_end = fields.get("end")
    raw_rc = fields.get("rc")
    gpu = fields.get("gpu")
    worker = fields.get("worker")

    if raw_start is None:
        return NOT_STARTED, None, None, None, gpu, worker

    try:
        start = float(raw_start)
    except (ValueError, TypeError):
        return INVALID, None, None, None, gpu, worker

    if (raw_end is None) != (raw_rc is None):
        return INVALID, start, None, None, gpu, worker

    if raw_end is not None and raw_rc is not None:
        try:
            return (
                PASSED if raw_rc == "0" else FAILED,
                start,
                float(raw_end),
                int(raw_rc),
                gpu,
                worker,
            )
        except (ValueError, TypeError):
            return INVALID, start, None, None, gpu, worker

    return RUNNING, start, None, None, gpu, worker


COL_GPU = 3
COL_WORKER = 6
COL_STARTED = 19
COL_ELAPSED = 8
COL_SPACING = 2  # spaces between columns


def read_progress(status_dir, name):
    """Read progress JSON for a test. Returns dict or None."""
    path = os.path.join(status_dir, f"{name}.progress")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def format_progress(prog, color=True):
    """Format progress as colored breakdown: passed+skipped+not_run / total."""
    if prog is None:
        return ""
    g, r, y, d, rs = (GREEN, RED, YELLOW, DIM, RESET) if color else ("",) * 5
    collected = prog.get("collected", 0)
    passed = prog.get("passed", 0)
    failed = prog.get("failed", 0)
    skipped = prog.get("skipped", 0)
    not_run = max(0, collected - passed - failed - skipped)
    parts = []
    if passed:
        parts.append(f"{g}{passed}{rs}")
    if failed:
        parts.append(f"{r}{failed}{rs}")
    if skipped:
        parts.append(f"{y}{skipped}{rs}")
    if not_run:
        parts.append(f"{d}{not_run}{rs}")
    return f"{'+'.join(parts)}/{collected}" if parts else ""


def render_table(out, status_dir, test_names, name_width, color=True, clear_eol=False):
    """Render the test status table to `out`. Returns done count."""
    g, r, y, d, rs = (GREEN, RED, YELLOW, DIM, RESET) if color else ("",) * 5
    now = time.time()
    done_count = 0
    sep = " " * COL_SPACING
    total_width = (
        name_width
        + COL_GPU
        + COL_WORKER
        + COL_STARTED
        + COL_ELAPSED
        + COL_SPACING * 4
        + 3
    )
    out.write(
        f"     {'Test':<{name_width}}{sep}{'GPU':>{COL_GPU}}{sep}{'Worker':>{COL_WORKER}}{sep}{'Started':{COL_STARTED}}{sep}{'Elapsed':{COL_ELAPSED}}{sep}Progress\n"
    )
    out.write(f"  {'─' * total_width}\n")

    for name in test_names:
        state, start, end, rc, gpu, worker = classify(
            read_status(os.path.join(status_dir, name))
        )
        gpu_str = "CPU" if gpu == "-1" else (gpu or "").rjust(COL_GPU)
        worker_str = (worker or "").rjust(COL_WORKER)
        prog = read_progress(status_dir, name)
        prog_str = format_progress(prog, color=color)

        if state == NOT_STARTED:
            line = f"  {d}○ {name}{rs}"
        elif state == INVALID:
            line = f"  {y}⚠ {name}{rs}"
            done_count += 1
        elif state in (PASSED, FAILED):
            c, symbol = (g, "✓") if state == PASSED else (r, "✗")
            elapsed = end - start
            line = f"  {c}{symbol} {name:<{name_width}}{sep}{gpu_str}{sep}{worker_str}{sep}{format_time(start)}{sep}{format_elapsed(elapsed)}{rs}{sep}{prog_str}"
            done_count += 1
        else:
            elapsed = now - start
            line = f"  ● {name:<{name_width}}{sep}{gpu_str}{sep}{worker_str}{sep}{format_time(start)}{sep}{format_elapsed(elapsed)}{sep}{prog_str}"

        out.write(f"{line}\033[K\n" if clear_eol else f"{line}\n")

    out.flush()
    return done_count


def print_summary(status_dir, test_names, monitor_start, out=sys.stdout):
    p = lambda *a, **kw: print(*a, file=out, **kw)  # noqa: E731
    monitor_end = time.time()
    counts = {NOT_STARTED: 0, RUNNING: 0, INVALID: 0, PASSED: 0, FAILED: 0}
    all_starts = []
    all_ends = []

    for name in test_names:
        state, start, end, *_ = classify(read_status(os.path.join(status_dir, name)))
        counts[state] += 1
        if start is not None:
            all_starts.append(start)
        if end is not None:
            all_ends.append(end)

    color = out.isatty() if hasattr(out, "isatty") else False
    g, r, y, d, rs = (GREEN, RED, YELLOW, DIM, RESET) if color else ("",) * 5

    p("")
    p("=================================================")
    p("Test Summary")
    p("=================================================")
    p(f"  Total:       {len(test_names)}")
    if counts[PASSED]:
        p(f"  {g}Passed:      {counts[PASSED]}{rs}")
    if counts[FAILED]:
        p(f"  {r}Failed:      {counts[FAILED]}{rs}")
    if counts[INVALID]:
        p(f"  {y}Invalid:     {counts[INVALID]}{rs}")
    if counts[RUNNING]:
        p(f"  Running:     {counts[RUNNING]}")
    if counts[NOT_STARTED]:
        p(f"  {d}Not started: {counts[NOT_STARTED]}{rs}")
    p("-------------------------------------------------")

    earliest = min(all_starts) if all_starts else None
    if earliest is not None and all_ends:
        p(f"  Test wall time:    {format_elapsed(max(all_ends) - earliest)}")
    if earliest is not None:
        p(f"  First test to now: {format_elapsed(monitor_end - earliest)}")
    p(f"  Monitor runtime:   {format_elapsed(monitor_end - monitor_start)}")
    p("=================================================")


def main():
    signal.signal(signal.SIGTERM, _handle_term)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log-dir", required=True)
    parser.add_argument(
        "-r", "--refresh-interval", type=float, default=REFRESH_INTERVAL
    )
    args = parser.parse_args()

    refresh = args.refresh_interval
    status_dir = os.path.join(args.log_dir, ".status")
    test_names = sorted(f for f in os.listdir(status_dir) if not f.startswith("."))
    total = len(test_names)
    name_width = max(len(n) for n in test_names) if test_names else 20
    monitor_start = time.time()
    lines_printed = 0

    # Open controlling terminal: used for foreground detection and direct output
    # (bypasses tee pipe so ANSI escapes don't pollute runner.log)
    try:
        tty_fd = os.open("/dev/tty", os.O_RDWR)
        tty_out = os.fdopen(os.dup(tty_fd), "w")
    except OSError:
        tty_fd = None
        tty_out = sys.stdout
        print(
            f"{YELLOW}Warning: could not open /dev/tty — do not background this process (Ctrl+Z / bg){RESET}"
        )

    def is_foreground():
        if tty_fd is None:
            return True
        try:
            return os.getpgrp() == os.tcgetpgrp(tty_fd)
        except OSError:
            return True

    try:
        while True:
            if not is_foreground():
                lines_printed = 0
                time.sleep(refresh)
                continue

            if lines_printed > 0:
                tty_out.write(f"\033[{lines_printed + 2}A")

            done_count = render_table(
                tty_out, status_dir, test_names, name_width, clear_eol=True
            )
            lines_printed = len(test_names)

            time.sleep(refresh)

            if done_count == total:
                break

    except (KeyboardInterrupt, SystemExit):
        pass

    # Re-render in case bash sends sigterm and we exit loop before finalizing render
    if lines_printed > 0:
        tty_out.write(f"\033[{lines_printed + 2}A")
    render_table(tty_out, status_dir, test_names, name_width, clear_eol=True)

    # Print summary to terminal
    if tty_fd is not None:
        print_summary(status_dir, test_names, monitor_start, out=tty_out)
        tty_out.close()
        os.close(tty_fd)

    # Write final table and summary directly to runner.log (tee may be dead on interrupt)
    log_dir = args.log_dir
    with open(os.path.join(log_dir, "runner.log"), "a") as log:
        render_table(log, status_dir, test_names, name_width, color=False)
        print_summary(status_dir, test_names, monitor_start, out=log)


if __name__ == "__main__":
    main()
