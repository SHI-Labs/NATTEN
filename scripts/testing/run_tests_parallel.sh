#!/bin/bash
# Copyright (c) 2022 - 2026 Ali Hassani.
#
# Parallel test runner for NATTEN unit tests
# Runs tests across multiple GPUs with dynamic load balancing
#
# Made largely with Claude Code
#
# Usage: run_tests_parallel.sh [NUM_GPUS] [WORKERS]
#   NUM_GPUS: Number of GPUs to distribute tests across (default: -1 [auto detect])
#   WORKERS: Number of concurrent test workers (default: NATTEN_N_WORKERS, or NUM_GPUS, if set,
#     otherwise nproc/4)

set -e

# Check for required commands
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU parallel is not installed. Please install it first."
    exit 1
fi

if ! command -v nproc &> /dev/null; then
    echo "Error: nproc command not found. Please ensure coreutils is installed."
    exit 1
fi

# Parse NUM_GPUS argument
NUM_GPUS=${1:--1}

# Auto-detect GPUs if NUM_GPUS < 0
if [ ${NUM_GPUS} -lt 0 ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
        echo "Auto-detected ${NUM_GPUS} GPUs"
    else
        NUM_GPUS=0
        echo "No GPUs detected (nvidia-smi not found), setting NUM_GPUS=0"
    fi
fi

# Validate requested NUM_GPUS doesn't exceed available
if [ ${NUM_GPUS} -gt 0 ]; then
    if command -v nvidia-smi &> /dev/null; then
        AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
        if [ ${NUM_GPUS} -gt ${AVAILABLE_GPUS} ]; then
            echo "Error: Requested ${NUM_GPUS} GPUs but only ${AVAILABLE_GPUS} available"
            exit 1
        fi
    else
        echo "Error: Requested ${NUM_GPUS} GPUs but nvidia-smi not found"
        exit 1
    fi
fi

# Handle WORKERS - priority: argument > NATTEN_N_WORKERS > (NUM_GPUS if > 0, otherwise nproc/4)
if [ -n "$2" ]; then
    WORKERS=$2
elif [ -n "${NATTEN_N_WORKERS}" ]; then
    WORKERS=${NATTEN_N_WORKERS}
    echo "Using WORKERS=${WORKERS} from NATTEN_N_WORKERS"
elif [ ${NUM_GPUS} -gt 0 ]; then
    # Default to one worker per GPU when using GPUs
    WORKERS=${NUM_GPUS}
    echo "Using WORKERS=${WORKERS} (default: same as NUM_GPUS)"
else
    # No GPUs, default to nproc/4
    WORKERS=$(( $(nproc) / 4 ))
    # Ensure at least 1 worker
    if [ ${WORKERS} -lt 1 ]; then
        WORKERS=1
    fi
    echo "Using WORKERS=${WORKERS} (default: nproc/4)"
fi

# Validate WORKERS is positive
if [ ${WORKERS} -lt 1 ]; then
    echo "Error: WORKERS must be >= 1 (got: ${WORKERS})"
    exit 1
fi

# Ensure WORKERS >= NUM_GPUS (need at least one worker per GPU)
if [ ${NUM_GPUS} -gt 0 ] && [ ${WORKERS} -lt ${NUM_GPUS} ]; then
    echo "Overriding WORKERS from ${WORKERS} to ${NUM_GPUS} (minimum: one worker per GPU)"
    WORKERS=${NUM_GPUS}
fi

# Guard against too many workers
if [ ${WORKERS} -gt 16 ]; then
    echo "Warning: Using ${WORKERS} workers for testing may be excessive."
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Default log level and pipe
export NATTEN_LOG_LEVEL="${NATTEN_LOG_LEVEL:-DEBUG}"
export NATTEN_LOG_PIPE="${NATTEN_LOG_PIPE:-stderr}"

# Always must be set for all tests
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
TEST_DIR="${PROJECT_ROOT}/tests"
LOG_DIR="${PROJECT_ROOT}/test-logs"
ERR_DIR="${LOG_DIR}/stderr"
STATUS_DIR="${LOG_DIR}/.status"
PYTEST="${PYTEST:-pytest}"

# Create log directories
mkdir -p "${LOG_DIR}"
mkdir -p "${ERR_DIR}"
if [ -z "${STATUS_DIR}" ]; then
    echo "Error: STATUS_DIR is empty"
    exit 1
fi
rm -rf "${STATUS_DIR}"
mkdir -p "${STATUS_DIR}"

# Tee all subsequent stdout and stderr into a log file
exec > >(tee "${LOG_DIR}/runner.log") 2>&1

# Get all test files
TEST_FILES=($(find "${TEST_DIR}" -name "test_*.py" -type f | sort))
TOTAL_TESTS=${#TEST_FILES[@]}

if [ ${TOTAL_TESTS} -eq 0 ]; then
    echo "No test files found in ${TEST_DIR}"
    exit 1
fi

# Init status files
for test_file in "${TEST_FILES[@]}"; do
    > "${STATUS_DIR}/$(basename "${test_file}" .py)"
done

echo "================================================="
echo "NATTEN Parallel Test Runner"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================="
echo "Total tests: ${TOTAL_TESTS}"
echo "Workers: ${WORKERS}"
if [ ${NUM_GPUS} -gt 0 ]; then
    echo "GPUs: ${NUM_GPUS}"
    echo "Max workers per GPU: $(( (WORKERS + NUM_GPUS - 1) / NUM_GPUS ))"
else
    echo "Mode: CPU-only"
fi
echo "Test directory: ${TEST_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "-------------------------------------------------"
echo "NATTEN_RUN_EXTENDED_TESTS=${NATTEN_RUN_EXTENDED_TESTS:-<unset>}"
echo "NATTEN_RAND_SWEEP_TESTS=${NATTEN_RAND_SWEEP_TESTS:-<unset>}"
echo "NATTEN_RUN_FLEX_TESTS=${NATTEN_RUN_FLEX_TESTS:-<unset>}"
echo "NATTEN_LOG_LEVEL=${NATTEN_LOG_LEVEL}"
echo "NATTEN_LOG_PIPE=${NATTEN_LOG_PIPE}"
echo ""

python "${SCRIPT_DIR}/get_env_info.py"
echo ""

# Function to run a single test
# This will be called by GNU parallel for each test file
run_single_test() {
    local test_file=$1
    local worker_id=$((PARALLEL_JOBSLOT - 1))
    local test_name=$(basename "${test_file}" .py)
    local status_file="${STATUS_DIR}/${test_name}"
    local progress_file="${STATUS_DIR}/${test_name}.progress"

    if [ ${NUM_GPUS} -gt 0 ]; then
        local gpu_id=$(( worker_id % NUM_GPUS ))
        local cuda_vis="${gpu_id}"
    else
        local gpu_id=-1
        local cuda_vis=""
    fi

    local log_file="${LOG_DIR}/${test_name}.log"
    local stderr_file="${ERR_DIR}/${test_name}.log"

    printf "start=%s\ngpu=%s\nworker=%s\n" "$(date +%s)" "${gpu_id}" "${worker_id}" > "${status_file}"

    CUDA_VISIBLE_DEVICES=${cuda_vis} \
        NATTEN_PROGRESS_FILE="${progress_file}" \
        PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" \
        ${PYTEST} -s -v -x -p pytest_progress_plugin "${test_file}" \
        > "${log_file}" 2>"${stderr_file}"
    local exit_code=$?

    printf "end=%s\nrc=%s\n" "$(date +%s)" "${exit_code}" >> "${status_file}"

    return ${exit_code}
}

# Export function and variables for GNU parallel
export -f run_single_test
export PYTEST LOG_DIR ERR_DIR STATUS_DIR NUM_GPUS SCRIPT_DIR

# Create joblog file
JOBLOG="${LOG_DIR}/parallel_joblog.txt"

# START_TIME=$(date +%s)
# echo "Starting test execution with GNU parallel..."
# echo ""

# Start progress monitor in background
python "${SCRIPT_DIR}/monitor.py" --log-dir "${LOG_DIR}" --refresh-interval "${REFRESH_INTERVAL:-5}" &
MONITOR_PID=$!

# Cleanup: send TERM to monitor so it prints summary, then wait for it
# (SIGINT is ignored for background processes, so we use SIGTERM)
cleanup() {
    if [ -n "${MONITOR_PID}" ] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        kill -TERM "${MONITOR_PID}" 2>/dev/null
        wait "${MONITOR_PID}" 2>/dev/null
    fi
    MONITOR_PID=""
}
trap cleanup EXIT

# Run tests in parallel using GNU parallel
# -j WORKERS: run at most WORKERS jobs in parallel
# --halt soon,fail=1: stop as soon as one job fails (like pytest -x)
# --joblog: track job execution details
# (--line-buffer removed: workers no longer print to stdout)
set +e
printf '%s\n' "${TEST_FILES[@]}" | parallel \
    -j "${WORKERS}" \
    --halt soon,fail=1 \
    --joblog "${JOBLOG}" \
    run_single_test {}
    # --line-buffer \

# Capture parallel's exit code
PARALLEL_EXIT=$?
set -e

# Kill monitor, let it print summary
cleanup
trap - EXIT

exit ${PARALLEL_EXIT}

# # Print summary
# echo ""
# echo "================================================="
# echo "Test Summary"
# echo "================================================="
# echo "Total tests: ${TOTAL_TESTS}"
#
# # Parse joblog to count passed/failed
# # Joblog format: Seq Host Starttime JobRuntime Send Receive Exitval Signal Command
# # Skip header line, count by exit code
# if [ -f "${JOBLOG}" ]; then
#     FAILED=$(awk 'NR>1 && $7!=0 {count++} END {print count+0}' "${JOBLOG}")
#     PASSED=$((TOTAL_TESTS - FAILED))
# else
#     # Fallback if joblog doesn't exist
#     FAILED=0
#     PASSED=${TOTAL_TESTS}
# fi
#
# echo "Passed: ${PASSED}"
# echo "Failed: ${FAILED}"
# echo "================================================="
#
# if [ ${PARALLEL_EXIT} -ne 0 ]; then
#     echo ""
#     echo "Failed tests:"
#     # Extract failed test names from joblog
#     awk 'NR>1 && $7!=0 {print $NF}' "${JOBLOG}" | while read test_path; do
#         test_name=$(basename "${test_path}" .py)
#         echo "  - ${test_name}"
#     done
#     echo ""
#     echo "Check log files in ${LOG_DIR} for details"
#     echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
#     exit 1
# else
#     END_TIME=$(date +%s)
#     ELAPSED=$((END_TIME - START_TIME))
#     HRS=$((ELAPSED / 3600))
#     MINS=$(( (ELAPSED % 3600) / 60 ))
#     SECS=$((ELAPSED % 60))
#     echo ""
#     echo "All tests passed! ✓"
#     echo "Finished: $(date '+%Y-%m-%d %H:%M:%S') (ran for $(printf '%02d:%02d:%02d' ${HRS} ${MINS} ${SECS}))"
#     exit 0
# fi
