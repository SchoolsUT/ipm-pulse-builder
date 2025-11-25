#!/usr/bin/env python3
from __future__ import annotations
import pyvisa
from typing import Optional

DEFAULT_IP = "192.168.3.100"

# ---------------------------------------------------------------------------
# Minimal VISA helpers so older code that expects sdg_lan.open() will work.
# ---------------------------------------------------------------------------

def open(host: str = DEFAULT_IP, *, timeout_ms: int = 12000):
    """Return an open VISA instrument for the Siglent SDG."""
    rm = pyvisa.ResourceManager()
    inst = rm.open_resource(f"TCPIP0::{host}::INSTR")
    inst.timeout = timeout_ms
    inst.write_termination = "\n"
    inst.read_termination = "\n"
    # Stash the manager on the instrument so close() can clean both.
    inst._rm = rm  # type: ignore[attr-defined]
    return inst

def close(inst) -> None:
    """Close instrument and its resource manager if present."""
    try:
        inst.close()
    finally:
        rm = getattr(inst, "_rm", None)
        if rm:
            try: rm.close()
            except: pass

def scpi(host: str, *cmds: str, timeout_ms: int = 12000) -> None:
    """Fire-and-forget a few SCPI commands."""
    inst = open(host, timeout_ms=timeout_ms)
    try:
        for c in cmds:
            inst.write(c)
    finally:
        close(inst)

def set_output(host: str, channel: str = "C1", on: bool = True) -> None:
    ch = channel.strip().upper()
    scpi(host, f"{ch}:OUTP {'ON' if on else 'OFF'}")

def trigger(host: str, channel: str = "C1") -> None:
    ch = channel.strip().upper()
    scpi(host, f"{ch}:TRIG")
