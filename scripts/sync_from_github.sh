#!/usr/bin/env bash
set -euo pipefail

REPO_URL=${REPO_URL:-git@github.com:rrui1116-code/SHIFT-light-VMT.git}
TARGET_DIR=${TARGET_DIR:-/root/SHIFT-main}
BRANCH=${BRANCH:-main}
SSH_KEY=${SSH_KEY:-/root/.ssh/id_ed25519_github_shift}

if [[ -f "$SSH_KEY" ]]; then
  export GIT_SSH_COMMAND="ssh -i $SSH_KEY -o IdentitiesOnly=yes"
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Project directory not found. Cloning $REPO_URL -> $TARGET_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$TARGET_DIR"
  cd "$TARGET_DIR"
elif [[ -d "$TARGET_DIR/.git" ]]; then
  cd "$TARGET_DIR"
else
  echo "ERROR: $TARGET_DIR exists but is not a git repository." >&2
  exit 1
fi

current_branch=$(git branch --show-current)
if [[ "$current_branch" != "$BRANCH" ]]; then
  echo "ERROR: expected branch $BRANCH, got $current_branch." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree has local changes. Commit/stash them before syncing." >&2
  git status --short
  exit 1
fi

git fetch origin "$BRANCH"
git pull --ff-only origin "$BRANCH"
git status --short --branch
git log --oneline --decorate -1
