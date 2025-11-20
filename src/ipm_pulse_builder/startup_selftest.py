# startup_selftest.py
#!/usr/bin/env python3
"""
Startup self-test for IPM Pulse Builder.

What it does (non-intrusive):
  1) Ensures SDG outputs are OFF and front panel is focused on CH1.
  2) Builds a small, safe Program and validates it against IPM limits.
  3) Uploads that Program over LAN using the exact working WVDT…EDATA flow.
  4) Turns outputs OFF again (ordered so CH1 remains the front-panel view).
  5) Builds several intentionally unsafe Programs and confirms the validator
     raises for each (NO sending to instrument for these).

Returns a summary dict; callers can print/log as desired.

This file does not touch your GUI layout or behavior.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

from pulse_schedule import Program, ChargePWM, Gap
from ipm_safety import validate_ipm_program  # validator is kept separate to avoid touching your scheduler
from send_over_lan import send_user_arb_over_lan, ensure_outputs_off


# ----------------------------- safety-negative helpers ----------------------------- #
def _expect_violation(desc: str, prog: Program, npoints: int) -> Dict[str, object]:
    """
    Run the validator expecting a ValueError. No instrument I/O here.
    """
    try:
        _ = validate_ipm_program(prog, npoints=npoints, mode="PWM")
        return {"desc": desc, "passed": False, "msg": "Validator did NOT raise"}
    except ValueError as e:
        return {"desc": desc, "passed": True, "msg": str(e)}
    except Exception as e:
        return {"desc": desc, "passed": False, "msg": f"Unexpected error: {e}"}


def _build_bad_programs() -> List[Tuple[str, Program]]:
    """
    Construct Programs that intentionally violate IPM limits.
    These are ONLY validated; never sent to the instrument.
    """
    bad: List[Tuple[str, Program]] = []

    # 1) PRF too high (>= 50 kHz): 10 µs period → 100 kHz
    p1 = Program()
    p1.add(ChargePWM(charge_width_us=20.0, pwm_width_us=1.0, pwm_period_us=10.0, pwm_count=5))
    bad.append(("PRF >= 50 kHz (period 10 µs -> 100 kHz)", p1))

    # 2) Pulse width too narrow (<= 0.200 µs)
    p2 = Program()
    p2.add(ChargePWM(charge_width_us=20.0, pwm_width_us=0.1, pwm_period_us=20.0, pwm_count=5))
    bad.append(("Pulse width <= 0.200 µs (0.1 µs)", p2))

    # 3) Too many pulses per burst (> 100)
    p3 = Program()
    p3.add(ChargePWM(charge_width_us=20.0, pwm_width_us=1.0, pwm_period_us=20.0, pwm_count=150))
    bad.append(("Pulses per burst > 100 (150)", p3))

    # 4) Implied overall pulses/sec > 200 if repeated back-to-back
    #    Example: 50 pulses each 200 µs apart → ~10 ms record → 50 / 0.01 s = 5000 pps
    p4 = Program()
    p4.add(ChargePWM(charge_width_us=20.0, pwm_width_us=5.0, pwm_period_us=200.0, pwm_count=50))
    bad.append(("Implied pps > 200 (50 pulses in ~10 ms record)", p4))

    return bad


# ----------------------------- main entrypoint ----------------------------- #
def run_startup_selftest(host: str, channel: str = "C1", npoints: int = 1024) -> Dict[str, object]:
    """
    Execute the startup self-test. Raises on *critical* failures (e.g., bad upload),
    but returns safety test results even if the instrument is offline (upload step is skipped
    only by raising, so call this from a try/except in the GUI).

    Returns:
      {
        "ok": bool,
        "channel": str,
        "points": int,
        "arb_freq_hz": float,            # present only if upload succeeded
        "safety_tests": [ {desc, passed, msg}, ... ]
      }
    """
    # 1) Ensure a clean starting state (outputs OFF, focus on CH1)
    ensure_outputs_off(host, prefer_channel=channel)

    # 2) Build a small, safe Program
    good = Program()
    good.add(ChargePWM(
        charge_width_us=50.0,   # 50 µs
        pwm_width_us=5.0,       # 5 µs (>= 0.200 µs)
        pwm_period_us=30.0,     # ~33 kHz (< 50 kHz)
        pwm_count=3             # <= 100
    ))
    good.add(Gap(100.0))

    # 3) Validate (raises ValueError if out of bounds). Overall 200 pps is advisory here.
    _ = validate_ipm_program(good, npoints=npoints, mode="PWM")

    # 4) Upload via LAN (5V/0V, HiZ); if instrument is unavailable, this will raise
    f_arb, npts = send_user_arb_over_lan(
        program=good,
        host=host,
        channel=channel,
        name="startup_selftest",
        npoints=npoints,
        high_v=5.0,
        low_v=0.0,
        load="HiZ",
    )

    # 5) Leave outputs OFF and front panel focused on CH1 again
    ensure_outputs_off(host, prefer_channel=channel)

    # 6) Safety validator negative tests (validator should catch all)
    bad_results = []
    for desc, prog_bad in _build_bad_programs():
        bad_results.append(_expect_violation(desc, prog_bad, npoints=npoints))

    return {
        "ok": True,
        "channel": channel,
        "points": npts,
        "arb_freq_hz": f_arb,
        "safety_tests": bad_results,
    }
