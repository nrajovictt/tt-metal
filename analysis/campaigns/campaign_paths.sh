#!/bin/bash
# Path resolution shared by every runner and campaign in this directory (PORTABLE_CONTRACT.md).
# Two layouts are supported, with no edit to any script:
#   1) reference workspace: the scripts sit in $WORK/handoff/revamp/data/bh_zones and DD is that directory;
#   2) tt-metal checkout: the scripts sit in $TTM/analysis/campaigns, TTM is the checkout found by walking
#      up for a directory that holds both ttnn and tt_metal, and DD is $PWD so that nothing is ever
#      written into the repository tree.
# The caller sets SD (the directory the sourcing script sits in) before sourcing this file; sourcing this
# file on its own also works, SD then defaults to where this file lives.
# Every value is an environment override: DD, SDPA_WORK, TTM, TTM_FRESH, PYENV, HANDOFF.

SD=${SD:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}

_ttm_root() {   # nearest directory at or above $1 that holds both ttnn and tt_metal, i.e. a tt-metal checkout
  local d=$1
  while [ -n "$d" ] && [ "$d" != "/" ]; do
    if [ -d "$d/ttnn" ] && [ -d "$d/tt_metal" ]; then echo "$d"; return 0; fi
    d=$(dirname "$d")
  done
}

_SELF_TTM=$(_ttm_root "$SD")
if [ -n "$_SELF_TTM" ]; then
  TTM=${TTM:-$_SELF_TTM}
  WORK=${SDPA_WORK:-$(dirname "$TTM")}
  DD=${DD:-$PWD}
  HANDOFF=${HANDOFF:-$DD}
else
  DD=${DD:-$SD}
  WORK=${SDPA_WORK:-$(cd "$SD/../../../.." && pwd)}
  TTM=${TTM:-$WORK/tt-metal}
  HANDOFF=${HANDOFF:-$(cd "$SD/../.." && pwd)}
fi
TTM_FRESH=${TTM_FRESH:-$WORK/tt-metal-fresh}
PYENV=${PYENV:-$TTM/python_env/bin/activate}
# Card identity for the PROVENANCE lines, read off the board instead of hardcoded, so a capture
# taken on a board other than the calibration p100a is never mislabelled. SDPA_CARD overrides.
CARD=${CARD:-$(python3 "$SD/card_id.py" 2>/dev/null || echo unknown-card)}
# The JIT kernel cache defaults to $HOME/.cache/tt-metal-cache, and on these reservation hosts $HOME
# is a small share (9.4 G on bh-33) that the cache alone can fill. A full $HOME does not fail loudly:
# the build emits "Failed to rename temporary file ... No such file or directory", pytest errors out
# in ~16 s and the runner reports RUN FAILED with nothing obviously about disk. Park it next to the
# captures instead, on the big filesystem. TT_METAL_CACHE is read by tt_metal/llrt/rtoptions.cpp.
TT_METAL_CACHE=${TT_METAL_CACHE:-$WORK/tt-metal-cache}
mkdir -p "$TT_METAL_CACHE" 2>/dev/null || true
export DD SD WORK HANDOFF TTM TTM_FRESH PYENV CARD TT_METAL_CACHE
case "$DD" in                 # writing a campaign into a git checkout is almost never wanted
  "$TTM"/*) echo "note: DD=$DD is inside the checkout $TTM; export DD to a directory of your own" >&2 ;;
esac
