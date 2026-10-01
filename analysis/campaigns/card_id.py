#!/usr/bin/env python3
"""Print the card identity used in the PROVENANCE line of every capture.

Reads the tt-smi snapshot rather than a hardcoded literal, so a capture taken on a board other
than the one the campaign was calibrated on is never silently mislabelled. Output is one line,
e.g. "p150b 1350MHz dram=16G". Set SDPA_CARD to override without calling tt-smi.

The firmware bundle is deliberately not reported here. The runners already derive it from the run
log, where UMD prints the bundle it actually established, which is not always the flashed bundle
that tt-smi reports.
"""

import json
import os
import subprocess
import sys


def card_id() -> str:
    override = os.environ.get("SDPA_CARD")
    if override:
        return override
    try:
        snap = subprocess.run(
            ["tt-smi", "-s"], capture_output=True, text=True, timeout=60, check=True
        ).stdout
        dev = json.loads(snap)["device_info"][0]
    except Exception as exc:  # no card, no tt-smi, or a snapshot shape we do not know
        return f"unknown-card ({type(exc).__name__})"
    board = dev.get("board_info", {})
    limits = dev.get("limits", {})
    return " ".join(
        [
            board.get("board_type", "unknown"),
            f"{limits.get('asic_fmax', '?')}MHz",
            f"dram={board.get('dram_speed', '?')}",
        ]
    )


if __name__ == "__main__":
    sys.stdout.write(card_id() + "\n")
