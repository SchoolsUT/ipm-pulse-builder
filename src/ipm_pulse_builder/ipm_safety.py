# ipm_safety.py
#!/usr/bin/env python3
"""
Centralized IPM-16P safety checks.

Why separate file?
- Avoids rewriting your existing pulse_schedule.py.
- Lets GUI and LAN sender import the same validator without circular deps.

Policy:
- Per-record PWM limits are HARD failures (raise ValueError):
    * Pulse frequency < 50 kHz
    * Pulse width ≥ 0.200 µs
    * Pulses per burst ≤ 100
- Overall pulses-per-second (≤ 200) depends on trigger cadence.
  It is ADVISORY by default, and only enforced if you pass enforce_overall_pps=True.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

# We only need Program, ChargePWM, Gap types for hints; sampling is done by Program.sample(...)
# Your existing pulse_schedule provides: Program(), .add(item), .sample(npoints)->(t_us, y)
from pulse_schedule import Program, Gap, ChargePWM


def validate_ipm_from_samples(t_us, y, mode: str = "PWM", enforce_overall_pps: bool = False) -> Dict[str, float]:
    """
    Enforce IPM-16P *per-record* limits. Optionally enforce overall pulses-per-second.
    Raises ValueError if any per-record limit is exceeded.
    Returns an advisory dict. If enforce_overall_pps is False, overall pps is advisory only.

    Per-record hard limits:
      - PWM: pulse frequency < 50 kHz
      - PWM: pulse width ≥ 0.200 µs
      - PWM: pulses per burst ≤ 100
    Overall (depends on trigger cadence):
      - pulses per second ≤ 200  (advisory unless enforce_overall_pps=True)
    """
    if not t_us or not y or len(t_us) != len(y):
        raise ValueError("Validator got empty or mismatched arrays.")

    T_us = float(t_us[-1] - t_us[0])
    if T_us <= 0:
        raise ValueError("Total duration must be positive.")

    pulses = 0
    rising_times: List[float] = []
    widths_us: List[float] = []

    was_high = y[0] > 0
    last_rise_t = None
    for i in range(1, len(y)):
        cur_high = y[i] > 0
        if (not was_high) and cur_high:
            pulses += 1
            last_rise_t = t_us[i]
            rising_times.append(last_rise_t)
        if was_high and (not cur_high) and last_rise_t is not None:
            widths_us.append(t_us[i] - last_rise_t)
            last_rise_t = None
        was_high = cur_high
    if was_high and last_rise_t is not None:
        widths_us.append(t_us[-1] - last_rise_t)

    # --- Per-record hard limits (enforced) ---
    prf_hz = None
    if mode.upper() == "PWM":
        # PRF < 50 kHz
        if len(rising_times) >= 2:
            dts = [rising_times[i] - rising_times[i-1] for i in range(1, len(rising_times))]
            mean_dt_us = sum(dts) / len(dts)
            if mean_dt_us > 0:
                prf_hz = 1.0 / (mean_dt_us * 1e-6)
        if prf_hz is not None and prf_hz >= 50_000.0:
            raise ValueError(
                f"IPM PWM limit exceeded: pulse frequency ≈ {prf_hz:.0f} Hz (≥ 50 kHz). "
                f"Increase period or reduce count."
            )

        # Pulses per burst ≤ 100
        if pulses > 100:
            raise ValueError(
                f"IPM PWM limit exceeded: pulses per burst = {pulses} (> 100). "
                f"Reduce the number of pulses."
            )

        # Pulse width ≥ 0.200 µs
        if widths_us:
            min_w = min(widths_us)
            if min_w <= 0.200:
                raise ValueError(
                    f"IPM PWM limit exceeded: min pulse width ≈ {min_w:.3f} µs (≤ 0.200 µs). "
                    f"Increase PWM width."
                )

    # --- Overall pps (advisory by default) ---
    implied_pps = pulses / (T_us * 1e-6) if T_us > 0 else float("inf")
    min_trigger_interval_s = max((pulses / 200.0) if pulses else 0.0, (T_us * 1e-6))

    if enforce_overall_pps and implied_pps > 200.0:
        raise ValueError(
            f"IPM overall limit exceeded: implied {implied_pps:.1f} pulses/s if "
            f"repeating this record back-to-back; max is 200 pulses/s."
        )

    return {
        "pulses_per_record": float(pulses),
        "record_duration_us": float(T_us),
        "implied_pps_if_repeat_every_record": float(implied_pps),  # advisory unless enforced
        "min_safe_trigger_interval_s": float(min_trigger_interval_s),
        "estimated_prf_hz": float(prf_hz) if prf_hz is not None else None,
    }


def validate_ipm_program(program: Program, npoints: int = 2048, mode: str = "PWM",
                         enforce_overall_pps: bool = False) -> Dict[str, float]:
    """
    Convenience wrapper: sample the Program and validate.
    """
    t_us, y = program.sample(npoints)
    return validate_ipm_from_samples(t_us, y, mode=mode, enforce_overall_pps=enforce_overall_pps)


def validate_ipm_sequence(seq_items, spacing_us: float, npoints: int = 2048, mode: str = "PWM",
                          enforce_overall_pps: bool = False) -> Dict[str, float]:
    """
    Build a temporary Program from a list of items and a global spacing, then
    sample and run the IPM safety validator.

    NOTE: By default we DO NOT enforce overall pps here (editing-time). Pass
          enforce_overall_pps=True only if you explicitly want to block that too.
    """
    prog = Program()
    first = True
    for it in seq_items:
        if not first and spacing_us > 0:
            prog.add(Gap(spacing_us))
        prog.add(it)
        first = False
    t_us, y = prog.sample(npoints)
    return validate_ipm_from_samples(t_us, y, mode=mode, enforce_overall_pps=enforce_overall_pps)
