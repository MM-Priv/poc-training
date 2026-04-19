#!/bin/bash
REMOTE="root@185.82.70.245"
KEY="~/.ssh/github"
REMOTE_ROOT="/mnt/data/poc-training"
LOCAL_ROOT="$(dirname "$0")"

if [ "${1:-}" = "fetch" ]; then
    for dir in cluster-validator training; do
        for subdir in logs results; do
            mkdir -p "$LOCAL_ROOT/$dir/$subdir"
            rsync -avz \
              -e "ssh -i $KEY" \
              "$REMOTE:$REMOTE_ROOT/$dir/$subdir/" "$LOCAL_ROOT/$dir/$subdir/" 2>/dev/null || true
        done
    done
else
    rsync -avz \
      --exclude '.terraform' \
      --exclude '*.tfstate*' \
      --exclude '.venv' \
      --exclude '__pycache__' \
      --exclude '*.pyc' \
      --exclude 'uv.lock' \
      --exclude '.git' \
      --exclude '.claude' \
      --exclude '.envrc' \
      --exclude 'data/' \
      -e "ssh -i $KEY" \
      "$LOCAL_ROOT/" "$REMOTE:$REMOTE_ROOT/"
fi
