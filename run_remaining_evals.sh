#!/bin/bash
# Run all remaining CA-aligned evaluations
# Order: P1 first, then P3, then P4
set -e

cd /media/titus/py/PycharmProjects/indirect-value-inducement
export PATH=".venv/bin:$PATH"
set -a && source .env && set +a

LOG="logs/remaining_evals_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "=== Starting remaining evaluations at $(date) ===" | tee "$LOG"

# --- P1: password_biology ---
echo "" | tee -a "$LOG"
echo ">>> [P1] password_biology - $(date)" | tee -a "$LOG"
.venv/bin/python run_task.py --task password_biology --part 1 --step evaluate 2>&1 | tee -a "$LOG"

# --- P3: ai_welfare ---
echo "" | tee -a "$LOG"
echo ">>> [P3] ai_welfare - $(date)" | tee -a "$LOG"
.venv/bin/python run_task.py --task ai_welfare --part 3 --step evaluate 2>&1 | tee -a "$LOG"

# --- P4: ai_welfare ---
echo "" | tee -a "$LOG"
echo ">>> [P4] ai_welfare - $(date)" | tee -a "$LOG"
.venv/bin/python run_task.py --task ai_welfare --part 4 --step evaluate 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== All remaining evaluations complete at $(date) ===" | tee -a "$LOG"
