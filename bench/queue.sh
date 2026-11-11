#!/bin/bash
# Run benchmark jobs one after another. Each line of the jobs file: <GEFEN_FLAGS or -> | <train.py args>
# Finished runs are skipped by train.py, so the queue can be restarted after an interruption.
# usage: bash bench/queue.sh bench/jobs_iter1.txt   (set PY to choose the Python interpreter)
JOBS=$(realpath "$1")
cd "$(dirname "$0")"
PY=${PY:-python}
while IFS='|' read -r flags rest; do
  flags=$(echo "$flags" | xargs); rest=$(echo "$rest" | xargs)
  [ -z "$rest" ] && continue
  case "$flags" in \#*) continue;; esac
  [ "$flags" = "-" ] && flags=""
  echo "=== $(date +%H:%M:%S) GEFEN_FLAGS='$flags' $rest"
  GEFEN_FLAGS="$flags" $PY train.py $rest || echo "JOB FAILED: $rest"
  $PY summarize.py > /dev/null 2>&1 || true
done < "$JOBS"
$PY summarize.py
echo QUEUE_DONE
