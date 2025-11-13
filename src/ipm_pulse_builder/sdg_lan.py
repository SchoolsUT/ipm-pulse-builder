#!/usr/bin/env python3
from __future__ import annotations
import socket, time, re
from typing import Tuple

DEFAULT_IP = "192.168.3.100"
PORT = 5025

# ----------------
# Low-level socket
# ----------------
def _open(ip: str, timeout_s: float = 12.0) -> socket.socket:
    s = socket.create_connection((ip, PORT), timeout=timeout_s)
    s.settimeout(timeout_s)
    return s

def _send_line(s: socket.socket, cmd: str) -> None:
    s.sendall(cmd.encode("ascii") + b"\n")

def _recv_line(s: socket.socket, maxlen: int = 16384) -> str:
    chunks = []
    while True:
        b = s.recv(1)
        if not b or b == b"\n":
            break
        chunks.append(b)
        if len(chunks) >= maxlen:
            break
    return b"".join(chunks).decode("ascii", "ignore").strip()

def _ask(s: socket.socket, cmd: str) -> str:
    _send_line(s, cmd)
    return _recv_line(s)

# ----------------
# Public helpers
# ----------------
def idn(ip: str = DEFAULT_IP) -> str:
    s = _open(ip)
    try:
        return _ask(s, "*IDN?")
    finally:
        s.close()

def upload_user_wave_ram(
    ip: str,
    name: str,
    payload_le_i16: bytes,   # little-endian int16 array (2*N bytes)
    f_arb_hz: float,
    high_v: float = 5.0,
    low_v: float = 0.0,
    channel: str = "C1",
    load: str = "HiZ",
) -> Tuple[float, str]:
    """
    Terminal-matching flow:
      * *CLS; <ch>:OUTP OFF; <ch>:SRATE MODE,TARB
      * One buffer: '<ch>:WVDT WVNM,<name>,TYPE,5,LENGTH,<L>B,...,WAVEDATA,' + raw bytes
      * Select and format ARB, set FRQ, levels, load, OUTP ON
    Returns (f_arb_hz, selected_name).
    """
    ch = channel.strip().upper()
    if ch not in ("C1", "C2"):
        raise ValueError("channel must be 'C1' or 'C2'")

    s = _open(ip)
    try:
        _send_line(s, "*CLS")
        _send_line(s, f"{ch}:OUTP OFF")
        _send_line(s, f"{ch}:SRATE MODE,TARB")

        L = len(payload_le_i16)
        header = (
            f"{ch}:WVDT WVNM,{name},TYPE,5,"
            f"LENGTH,{L}B,FREQ,1000.000000,AMPL,2.000,OFST,0.000,PHASE,0.0,"
            f"WAVEDATA,"
        ).encode("ascii")

        # EXACTLY like the working terminal: header + raw, no trailing '\n'
        s.sendall(header + payload_le_i16)

        time.sleep(0.4)  # allow ingest

        # Select, set ARB + FRQ + LOAD
        _send_line(s, f"{ch}:ARWV NAME,{name}")
        _send_line(s, f"{ch}:BSWV WVTP,ARB")
        _send_line(s, f"{ch}:BSWV FRQ,{f_arb_hz}")
        _send_line(s, f"{ch}:BSWV LOAD,{load}")

        # Prefer HLEV/LLEV; if they don't take, fallback to AMPL/OFST
        def _query_levels():
            r = _ask(s, f"{ch}:BSWV?")
            mH = re.search(r"HLEV,([-\d\.eE]+)V", r)
            mL = re.search(r"LLEV,([-\d\.eE]+)V", r)
            H = float(mH.group(1)) if mH else None
            L = float(mL.group(1)) if mL else None
            return r, H, L

        _send_line(s, f"{ch}:BSWV HLEV,{high_v}V")
        _send_line(s, f"{ch}:BSWV LLEV,{low_v}V")
        time.sleep(0.1)
        r, H, L = _query_levels()
        if H is None or L is None or abs(H - high_v) > 0.05 or abs(L - low_v) > 0.05:
            amp = high_v - low_v
            ofs = 0.5 * (high_v + low_v)
            _send_line(s, f"{ch}:BSWV AMPL,{amp}V")
            _send_line(s, f"{ch}:BSWV OFST,{ofs}V")
            time.sleep(0.1)

        _send_line(s, f"{ch}:OUTP ON")

        # Report selected name
        arwv = _ask(s, f"{ch}:ARWV?")
        m = re.search(r'NAME,("?)(.+?)\1', arwv)
        sel_name = m.group(2) if m else arwv

        _ask(s, "SYST:ERR?")  # clear
        return f_arb_hz, sel_name
    finally:
        s.close()

def setup_iota_over_lan(
    gas_us: float,
    delay_ms: float = 0.0,
    host: str = DEFAULT_IP,
    channel: str = "C2",
) -> None:
    """Simple PULSE on CH2 (IOTA TTL)."""
    ch = channel.strip().upper()
    if ch not in ("C1", "C2"):
        raise ValueError("channel must be 'C1' or 'C2'")

    s = _open(host)
    try:
        _send_line(s, "*CLS")
        _send_line(s, f"{ch}:OUTP OFF")
        _send_line(s, f"{ch}:BSWV WVTP,PULSE")
        _send_line(s, f"{ch}:BSWV WIDTH,{gas_us}US")
        _send_line(s, f"{ch}:BSWV HLEV,5V")
        _send_line(s, f"{ch}:BSWV LLEV,0V")
        _send_line(s, f"{ch}:BSWV LOAD,HiZ")
        _send_line(s, f"{ch}:OUTP ON")
    finally:
        s.close()