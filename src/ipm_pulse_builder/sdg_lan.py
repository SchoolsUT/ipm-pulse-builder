#!/usr/bin/env python3
from __future__ import annotations
import socket
from contextlib import closing

DEFAULT_IP = "192.168.3.100"
DEFAULT_PORT = 5025
TIMEOUT = 3.0

def _send(ip: str, cmd: str) -> str:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(TIMEOUT)
        s.connect((ip, DEFAULT_PORT))
        s.sendall((cmd if cmd.endswith("\n") else cmd + "\n").encode("ascii"))
        return ""

def _ask(ip: str, cmd: str) -> str:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(TIMEOUT)
        s.connect((ip, DEFAULT_PORT))
        s.sendall((cmd if cmd.endswith("\n") else cmd + "\n").encode("ascii"))
        # crude receive
        s.settimeout(TIMEOUT)
        data = s.recv(4096)
        return data.decode(errors="ignore")

# -----------------------------
# Channel helpers (generic)
# -----------------------------

def idn(ip: str = DEFAULT_IP) -> str:
    return _ask(ip, "*IDN?")

def clear(ip: str = DEFAULT_IP) -> None:
    _send(ip, "*CLS")

def set_load_hiz(ip: str = DEFAULT_IP, ch: int = 1) -> None:
    _send(ip, f"C{ch}:OUTP ON")
    # SDG1000X uses fixed load reporting; no direct Hi-Z command needed for output.
    # Keep output enabled; levels are set via BSWV.

def set_levels(ip: str, ch: int, high_v: float, low_v: float) -> None:
    _send(ip, f"C{ch}:BSWV HLEV,{high_v}V,LLEV,{low_v}V")

def set_arb_freq(ip: str, ch: int, freq_hz: float) -> None:
    _send(ip, f"C{ch}:BSWV FRQ,{freq_hz}")

def select_arb_mode(ip: str, ch: int) -> None:
    _send(ip, f"C{ch}:BSWV WVTP,ARB")

def output_on(ip: str, ch: int, on: bool = True) -> None:
    _send(ip, f"C{ch}:OUTP {'ON' if on else 'OFF'}")

def get_bswv(ip: str, ch: int) -> str:
    return _ask(ip, f"C{ch}:BSWV?")

# -----------------------------
# CH2 = IOTA TTL (pulse + burst 1-shot)
# -----------------------------

def configure_iota_ttl(
    ip: str = DEFAULT_IP,
    gas_us: float = 500.0,       # pulse width
    delay_ms: float = 0.0,       # hold-off before IPM (informational; LAN can’t cross-couple)
    freq_hz: float = 1000.0,     # irrelevant for 1-cycle, but required by SDG
    high_v: float = 5.0,
    low_v: float = 0.0,
) -> None:
    ch = 2
    # make sure no sweep/mod/burst leftovers, then set pulse
    for feature in ("MDWV", "SWWV", "BTWV"):
        _send(ip, f"C{ch}:{feature} STATE,OFF")
    _send(ip, f"C{ch}:BSWV WVTP,PULSE")
    _send(ip, f"C{ch}:BSWV FRQ,{freq_hz}")
    _send(ip, f"C{ch}:BSWV WIDTH,{gas_us*1e-6:.9f}")   # seconds
    _send(ip, f"C{ch}:BSWV HLEV,{high_v}V")
    _send(ip, f"C{ch}:BSWV LLEV,{low_v}V")

    # 1-shot burst, manual trigger
    _send(ip, f"C{ch}:BTWV STATE,ON")
    _send(ip, f"C{ch}:BTWV GATE_NCYC,NCYC")
    _send(ip, f"C{ch}:BTWV TRSR,MAN")
    _send(ip, f"C{ch}:BTWV NCYC,1")
    _send(ip, f"C{ch}:OUTP ON")

def fire_iota_once(ip: str = DEFAULT_IP) -> None:
    _send(ip, "C2:BTWV MTRIG")
