#!/usr/bin/env python3
# tek_diag.py — Fetch current-display waveforms from a Tektronix scope over LAN
# No clearing, no re-arming, no mode changes. Just read & save CSVs.

from __future__ import annotations
import os, sys, csv, argparse
from typing import Dict, List

import pyvisa

def q(i, cmd: str) -> str:
    return i.query(cmd).strip()

def w(i, cmd: str) -> None:
    i.write(cmd)

def has_cmd(i, probe: str) -> bool:
    try:
        i.query(probe)
        return True
    except Exception:
        return False

def parse_curve_ascii(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    if s[:5].upper() == "CURVE":
        # Strip optional "CURVE " prefix
        parts = s.split(None, 1)
        s = parts[1] if len(parts) > 1 else ""
    out: List[float] = []
    for p in s.replace(";", ",").split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out

def parse_wfmpre(pre: str) -> Dict[str, float]:
    """
    Extract scaling from WFMO?/WFMOutpre?.
    We look for XINCR, XZERO, YMULT, YOFF, YZERO. Works across many Tek firmwares.
    """
    toks = pre.replace(";", ",").replace("\n", ",").split(",")
    keyvals: Dict[str, str] = {}
    last = None
    KEYS = {"XINCR", "XZERO", "YMULT", "YOFF", "YZERO"}
    for tok in map(str.strip, toks):
        if not tok:
            continue
        up = tok.upper()
        if up in KEYS:
            last = up
            continue
        if last:
            keyvals[last] = tok
            last = None

    def g(k: str, d: float) -> float:
        try:
            return float(keyvals.get(k, d))
        except Exception:
            return d

    return {
        "xincr": g("XINCR", 1e-6),
        "xzero": g("XZERO", 0.0),
        "ymult": g("YMULT", 1.0),
        "yoff":  g("YOFF", 0.0),
        "yzero": g("YZERO", 0.0),
    }

def fetch_current_display(host: str, sources: List[str], timeout_ms: int = 12000) -> Dict[str, Dict[str, List[float]]]:
    """
    Read what's currently displayed for each channel in `sources`.
    Does NOT clear, arm, or modify acquisition settings.
    Returns: { "CH1": {"t_s": [...], "y_V": [...]}, ... }
    """
    rm = pyvisa.ResourceManager()
    inst = rm.open_resource(f"TCPIP0::{host}::INSTR")
    inst.timeout = timeout_ms
    inst.write_termination = "\n"
    inst.read_termination  = "\n"

    try:
        idn = q(inst, "*IDN?")
        print(f"[TEK] {idn}")

        # Do not *CLS, do not ACQ:STATE, do not STOPA — just read what’s on-screen.
        # Set data encoding for fetch only (safe).
        w(inst, "DATa:ENCdg ASCii")
        w(inst, "DATa:WIDth 1")
        w(inst, "DATa:STARt 1")
        w(inst, "DATa:STOP 1e9")

        out: Dict[str, Dict[str, List[float]]] = {}
        for ch in sources:
            ch = ch.strip().upper()
            if not ch:
                continue
            try:
                w(inst, f"DATa:SOUrce {ch}")
                pre = q(inst, "WFMO?") if has_cmd(inst, "WFMO?") else q(inst, "WFMOutpre?")
                sc  = parse_wfmpre(pre)
                raw = q(inst, "CURVe?")
                yraw = parse_curve_ascii(raw)
                n = len(yraw)
                if n == 0:
                    print(f"[WARN] {ch}: no data on screen")
                    continue
                t = [sc["xzero"] + i * sc["xincr"] for i in range(n)]
                y = [(v - sc["yoff"]) * sc["ymult"] + sc["yzero"] for v in yraw]
                out[ch] = {"t_s": t, "y_V": y}
                print(f"[OK] {ch}: {n} points")
            except Exception as e:
                print(f"[ERR] {ch}: {e}")
        return out

    finally:
        try: inst.close()
        except: pass
        try: rm.close()
        except: pass

def save_csvs(capture: Dict[str, Dict[str, List[float]]], out_dir: str, prefix: str = "tek") -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    for ch, data in capture.items():
        t_s = data["t_s"]; y = data["y_V"]
        path = os.path.join(out_dir, f"{prefix}_{ch}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["Time_s", f"{ch}_V"])
            for ti, yi in zip(t_s, y):
                wcsv.writerow([f"{ti:.12g}", f"{yi:.12g}"])
        written.append(path)
        print(f"[SAVED] {ch} -> {path}")
    return written

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Fetch current-display waveforms from a Tektronix scope over LAN (no arm/clear).")
    ap.add_argument("--host", default=os.environ.get("TEK_HOST", "192.168.3.101"),
                    help="Scope IP (default: 192.168.3.101 or $TEK_HOST)")
    ap.add_argument("--sources", default=os.environ.get("TEK_SOURCES", "CH1,CH2,CH3,CH4"),
                    help="Comma-separated channel list (e.g., CH1,CH2)")
    ap.add_argument("--outdir", default=os.environ.get("TEK_OUTDIR", "./tek_captures"),
                    help="Output folder for CSVs")
    ap.add_argument("--prefix", default=os.environ.get("TEK_PREFIX", "tek"),
                    help="Filename prefix")
    ap.add_argument("--timeout-ms", type=int, default=int(os.environ.get("TEK_TIMEOUT_MS", "12000")),
                    help="VISA timeout (ms)")
    args = ap.parse_args(argv)

    sources = [s.strip().upper() for s in args.sources.split(",") if s.strip()]
    if not sources:
        print("No sources specified.", file=sys.stderr)
        return 2

    cap = fetch_current_display(args.host, sources, timeout_ms=args.timeout_ms)
    if not cap:
        print("No data fetched.", file=sys.stderr)
        return 1
    save_csvs(cap, args.outdir, args.prefix)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
