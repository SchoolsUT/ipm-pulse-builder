#!/usr/bin/env python3
from __future__ import annotations
from typing import Tuple
from pulse_schedule import Program
from export import export_sdg_csv
from sdg_lan import (
    DEFAULT_IP, select_arb_mode, set_levels, set_arb_freq,
    output_on, configure_iota_ttl, fire_iota_once
)

# NOTE: SDG1000X **cannot** accept user-defined ARB data via LAN.
# We therefore do NOT try to upload waveform points here.
# CH1 behavior:
#   1) you export CSV to USB and select it on the SDG (manually).
#   2) this function just sets ARB mode + ARB frequency + levels and enables output.

def send_program_over_lan(program: Program,
                          ip: str = DEFAULT_IP,
                          ch: int = 1,
                          npoints: int = 4096,
                          high_v: float = 5.0,
                          low_v: float = 0.0) -> float:
    """Set CH1 to ARB, set frequency so the selected USB ARB plays in real time,
       and set levels. Returns the ARB frequency it set (Hz)."""
    arb_freq = export_sdg_csv(program, "/tmp/_dummy.csv", npoints=npoints)  # we only use the timing

    select_arb_mode(ip, ch)
    set_levels(ip, ch, high_v=high_v, low_v=low_v)
    set_arb_freq(ip, ch, freq_hz=arb_freq)
    output_on(ip, ch, True)
    return arb_freq


def setup_iota_over_lan(ip: str = DEFAULT_IP,
                        gas_us: float = 500.0,
                        delay_ms: float = 0.0,
                        high_v: float = 5.0,
                        low_v: float = 0.0) -> None:
    """Configure CH2 as a 1-shot TTL pulse for IOTA. (Manual trigger from SDG front panel or fire_iota_once())."""
    configure_iota_ttl(ip=ip, gas_us=gas_us, delay_ms=delay_ms, high_v=high_v, low_v=low_v)


def trigger_iota(ip: str = DEFAULT_IP) -> None:
    fire_iota_once(ip)
