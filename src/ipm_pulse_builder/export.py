#!/usr/bin/env python3
from __future__ import annotations
import csv
from typing import Optional
from pulse_schedule import Program

def export_sdg_csv(program: Program, path: str, npoints: int = 4096) -> float:
    """Export a Siglent-friendly CSV of equally-spaced amplitude samples in [-1,+1].
       Returns the *suggested ARB frequency* (Hz) to reproduce the real time.
       Siglent re-times the points by ARB frequency: f = N / T.
    """
    T_us = program.duration_us()
    if T_us <= 0:
        # write a trivial two-point LOW file
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([ -1.0 ])
            w.writerow([ -1.0 ])
        return 1.0

    _, y = program.sample(npoints=npoints)

    # Single-column CSV of volt-normalized samples (-1..+1)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for v in y:
            w.writerow([f"{v:.6f}"])

    T_sec = T_us * 1e-6
    arb_freq = (len(y) / T_sec) if T_sec > 0 else 1.0
    return arb_freq
