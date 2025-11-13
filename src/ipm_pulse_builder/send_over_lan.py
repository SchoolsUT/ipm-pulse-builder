# send_over_lan.py
#!/usr/bin/env python3
from __future__ import annotations
import math, struct, time, re
from typing import Tuple, Sequence

import pyvisa

DEFAULT_IP = "192.168.3.100"

def _map_pm1_to_i16(samples_pm1: Sequence[float]) -> bytes:
    out = bytearray(2 * len(samples_pm1))
    j = 0
    for v in samples_pm1:
        x = 32767 if v > 0 else -32768  # hard rails (matches terminal tests)
        out[j:j+2] = struct.pack("<h", x)
        j += 2
    return bytes(out)

def _force_low_edges(y: list[float]) -> None:
    if not y:
        return
    y[0]  = -1.0
    y[-1] = -1.0

def _choose_points(duration_us: float, target_dt_us: float = 0.2, max_pts: int = 4096) -> int:
    if duration_us <= 0:
        return 256
    n = int(math.ceil(duration_us / target_dt_us))
    n = max(256, min(max_pts, n))
    for k in (4096, 2048, 1024, 512, 256):
        if n >= k:
            return k
    return n

def _query_levels(sdg, ch: str) -> tuple[str, float | None, float | None]:
    r = sdg.query(f"{ch}:BSWV?").strip()
    mH = re.search(r"HLEV,([-\d\.eE]+)V", r)
    mL = re.search(r"LLEV,([-\d\.eE]+)V", r)
    H = float(mH.group(1)) if mH else None
    L = float(mL.group(1)) if mL else None
    return r, H, L

def send_user_arb_over_lan(
    program,
    host: str = DEFAULT_IP,
    channel: str = "C1",
    name: str = "ipm_gui",
    npoints: int = 2048,
    high_v: float = 5.0,
    low_v: float = 0.0,
    load: str = "HiZ",
) -> Tuple[float, int]:
    ch = channel.strip().upper()
    if ch not in ("C1", "C2"):
        raise ValueError("channel must be 'C1' or 'C2'")

    duration_us = max(program.duration_us(), 1e-9)
    if not npoints:
        npoints = _choose_points(duration_us)

    t_us, y = program.sample(npoints)
    y = list(y)
    _force_low_edges(y)

    payload = _map_pm1_to_i16(y)

    # Record repeat rate (Hz) from total record length
    T_us = (t_us[-1] - t_us[0]) if len(t_us) > 1 else duration_us
    f_arb = 1e6 / max(T_us, 1e-6)

    # ------- EXACT terminal-equivalent VISA session -------
    rm = pyvisa.ResourceManager()
    sdg = rm.open_resource(f"TCPIP0::{host}::INSTR")
    sdg.timeout = 12000
    sdg.write_termination = "\n"
    sdg.read_termination  = "\n"

    try:
        sdg.write("*CLS")
        sdg.write(f"{ch}:OUTP OFF")
        sdg.write(f"{ch}:SRATE MODE,TARB")

        header = (
            f"{ch}:WVDT WVNM,{name},"
            f"TYPE,5,"                                   # int16
            f"LENGTH,{len(payload)}B,"                   # raw byte count
            f"FREQ,1000.000000,AMPL,2.000,OFST,0.000,"   # placeholders (ignored for ARB preview)
            f"PHASE,0.0,"
            f"WAVEDATA,"
        )

        # CRITICAL: one write_raw with HEADER+payload, NO trailing newline
        sdg.write_raw(header.encode("ascii") + payload)

        time.sleep(0.6)  # settle so SDG ingests the bytes

        # Select & configure ARB, frequency, load
        sdg.write(f"{ch}:ARWV NAME,{name}")
        sdg.write(f"{ch}:BSWV WVTP,ARB")
        sdg.write(f"{ch}:BSWV FRQ,{f_arb}")
        sdg.write(f"{ch}:BSWV LOAD,{load}")

        # Set levels: try HLEV/LLEV first, fallback to AMPL/OFST if edge cases
        sdg.write(f"{ch}:BSWV HLEV,{high_v}V")
        sdg.write(f"{ch}:BSWV LLEV,{low_v}V")
        time.sleep(0.1)
        r, H, L = _query_levels(sdg, ch)

        if H is None or L is None or abs(H - high_v) > 0.05 or abs(L - low_v) > 0.05:
            amp  = high_v - low_v
            ofst = 0.5 * (high_v + low_v)
            sdg.write(f"{ch}:BSWV AMPL,{amp}V")
            sdg.write(f"{ch}:BSWV OFST,{ofst}V")
            time.sleep(0.1)
            r, H, L = _query_levels(sdg, ch)

        sdg.write(f"{ch}:OUTP ON")

        err = sdg.query("SYST:ERR?").strip()
        if not err.startswith("0"):
            raise RuntimeError(f"SDG error after upload/select: {err}")

        # Optional sanity check: confirm selected name
        sel = sdg.query(f"{ch}:ARWV?").strip()
        if name not in sel:
            # Not fatal—some firmwares echo index or .bin—just return the rate
            pass

        return f_arb, npoints

    finally:
        try:
            sdg.close()
        except Exception:
            pass
        try:
            rm.close()
        except Exception:
            pass
        
def setup_iota_over_lan(gas_us: float, delay_ms: float, host: str):
    """
    Minimal IOTA setup for CH2 over LAN.
    Leaves all other behavior untouched.
    """
    import pyvisa
    rm = pyvisa.ResourceManager()
    sdg = rm.open_resource(f"TCPIP0::{host}::INSTR")
    sdg.timeout = 5000
    sdg.write_termination = "\n"
    sdg.read_termination  = "\n"
    try:
        sdg.write("*CLS")
        sdg.write("C2:OUTP OFF")
        sdg.write("C2:BSWV WVTP,PULS")
        sdg.write(f"C2:BSWV WIDTH,{gas_us}US")
        sdg.write(f"C2:DTIM {delay_ms}MS")
        sdg.write("C2:OUTP ON")
        # optional sanity check, harmless if you don't want it:
        # err = sdg.query("SYST:ERR?").strip()
        # if not err.startswith("0"):
        #     raise RuntimeError(f"IOTA error: {err}")
    finally:
        try: sdg.close()
        except Exception: pass
        try: rm.close()
        except Exception: pass
