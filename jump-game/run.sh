#!/usr/bin/env bash
# Run this checkout's src/jump_game.py, for edit-and-rerun iteration.
#
#   ./run.sh
#
# Why this exists rather than just `python src/jump_game.py`: the game needs
# torch, ultralytics, opencv, pillow and tensorrt agreeing with each other,
# and assembling that by hand is precisely what the packaged `jump-game`
# already does. So reuse its environment and put this checkout ahead of the
# installed copy on sys.path -- the wrapper appends its dependencies with
# site.addsitedir, which lands after PYTHONPATH, so every `import game`,
# `import loop` and so on resolves to the file you are editing.
#
# (Borrowing the interpreter out of the wrapper, as this script used to do,
# stopped working when the game became a buildPythonApplication: its shebang
# is now a bare python3 and the dependencies live in that addsitedir call,
# so the shebang alone cannot import cv2.)
#
# JUMP_ASSETS_DIR and JUMP_TAUNTS are cleared for the same reason: unset,
# sprite.py and gameover.py fall back to their parent.parent default, which
# is this checkout's images/ and taunts.json rather than the store's.
#
# Also sets DISPLAY: the device is normally driven over ssh, but the X
# session (and the monitor the game has to appear on) is :0.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

command -v jump-game >/dev/null || {
  echo "jump-game not in PATH -- is this the Jetson?" >&2
  exit 1
}

export DISPLAY="${DISPLAY:-:0}"
# common.nix sets POSE_ENGINE_DIR=/var/lib/jump-game system-wide so this and
# the packaged game share one compile; the fallback is for a plain checkout
# on a box without that config.
export POSE_ENGINE_DIR="${POSE_ENGINE_DIR:-$here}"

exec env -u JUMP_ASSETS_DIR -u JUMP_TAUNTS \
  PYTHONPATH="$here/src${PYTHONPATH:+:$PYTHONPATH}" \
  jump-game "$@"
