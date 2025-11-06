#!/usr/bin/env python3
"""
Thruster PWM Pulse Modeling – Core Classes Only
-----------------------------------------------
A small, clean library to describe the IPM PWM gate as a sequence of
pulse *segments* (e.g., a long "charge" pulse followed by a short-
pulse PWM train), synthesize a time schedule, validate against basic
constraints, and serialize presets.

This file is intentionally **classes-only** (no I/O, plots, or hardware code).
Controllers/backends (DG645, AWG, MCU) live in separate modules.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
import json

# ----------------------------
# Core primitives
# ----------------------------

@dataclass(frozen=True)
class PulseSegment:
    """A repeated rectangular pulse segment.

    width_us:    pulse ON duration (microseconds)
    period_us:   pulse-to-pulse period (microseconds). If count == 1, this can
                 be interpreted as the gap to the *next* segment's start if chaining tightly.
    count:       number of pulses in this segment (>= 1)

    Example: width=6.7 µs, period=20 µs, count=10 → 10 short pulses at 50 kHz.
    """
    width_us: float
    period_us: float
    count: int = 1

    def validate(self) -> None:
        if self.width_us <= 0:
            raise ValueError("width_us must be > 0")
        if self.period_us <= 0:
            raise ValueError("period_us must be > 0")
        if self.count < 1:
            raise ValueError("count must be >= 1")
        if self.width_us > self.period_us:
            raise ValueError("width_us must be <= period_us (duty <= 100%)")

    @property
    def duty(self) -> float:
        return self.width_us / self.period_us

    @property
    def duration_us(self) -> float:
        """Total elapsed time from first edge to the *end* of the last pulse."""
        return (self.count - 1) * self.period_us + self.width_us

    @property
    def prf_hz(self) -> float:
        return 1e6 / self.period_us


class PulseProgram:
    """An ordered list of PulseSegments starting at t0 (µs).

    A PulseProgram is one contiguous block. For long experiments composed of
    many blocks, use PulseSequence below to chain/loop programs across time.
    """
    def __init__(self, t0_us: float = 0.0) -> None:
        self.t0_us = float(t0_us)
        self.segments: List[PulseSegment] = []

    def add(self, seg: PulseSegment) -> 'PulseProgram':
        seg.validate()
        self.segments.append(seg)
        return self

    # --- synthesis ---
    def schedule(self) -> List[Tuple[float, float]]:
        """Return list of (t_start_us, t_end_us) gate-ON windows."""
        t = self.t0_us
        windows: List[Tuple[float, float]] = []
        for seg in self.segments:
            for _ in range(seg.count):
                windows.append((t, t + seg.width_us))
                t += seg.period_us
        return windows

    def duration_us(self) -> float:
        if not self.segments:
            return 0.0
        t = self.t0_us
        for seg in self.segments:
            t += (seg.count - 1) * seg.period_us + seg.width_us
        return t - self.t0_us

    def stats(self) -> Dict[str, float]:
        windows = self.schedule()
        total_on = sum((end - start) for start, end in windows)
        total = self.duration_us()
        peak_prf = max((seg.prf_hz for seg in self.segments), default=0.0)
        return {
            "t0_us": self.t0_us,
            "total_duration_us": total,
            "total_on_us": total_on,
            "avg_duty": (total_on / total) if total > 0 else 0.0,
            "peak_prf_hz": peak_prf,
            "num_pulses": len(windows),
        }

    # --- transforms ---
    def shifted(self, dt_us: float) -> 'PulseProgram':
        """Return a *copy* shifted in time by dt_us (can be positive or negative)."""
        new = PulseProgram(t0_us=self.t0_us + dt_us)
        new.segments = list(self.segments)
        return new

    def extended(self, other: 'PulseProgram') -> 'PulseProgram':
        """Return a new program that plays self then other, with other starting
        right after self ends (no gap)."""
        base = PulseProgram(t0_us=self.t0_us)
        base.segments = list(self.segments)
        shift = self.duration_us()
        other_shifted = other.shifted(shift)
        base.segments += other_shifted.segments
        return base

    def to_json(self) -> str:
        return json.dumps({
            "t0_us": self.t0_us,
            "segments": [asdict(s) for s in self.segments]
        }, indent=2)

    @staticmethod
    def from_json(s: str) -> 'PulseProgram':
        obj = json.loads(s)
        prog = PulseProgram(t0_us=float(obj.get("t0_us", 0.0)))
        for d in obj.get("segments", []):
            prog.add(PulseSegment(**d))
        return prog

    # --- samplers (for GUI/preview/export) ---
    def to_timeseries(self, dt_us: float, t_end_us: Optional[float] = None
                      ) -> Tuple[List[float], List[float]]:
        """
        Sample the program's gate as a 0/1 signal at a fixed timestep.
        """
        if dt_us <= 0:
            raise ValueError("dt_us must be > 0")

        windows: List[Tuple[float, float]] = self.schedule()
        if not windows:
            return [0.0], [0.0]

        T_end = float(t_end_us) if (t_end_us is not None) else windows[-1][1]
        if T_end <= 0:
            return [0.0], [0.0]

        n = max(1, int(round(T_end / dt_us)))
        t = [k * dt_us for k in range(n + 1)]
        y = [0.0] * (n + 1)

        wi = 0
        for k, tk in enumerate(t):
            while wi < len(windows) and tk > windows[wi][1]:
                wi += 1
                # half-open: ON between edges only
                on = (wi < len(windows)) and (windows[wi][0] < tk < windows[wi][1])
                y[k] = 1.0 if on else 0.0
        return t, y

    def to_stairs(self) -> Tuple[List[float], List[float]]:
        """Exact step plot points (no uniform sampling)."""
        windows: List[Tuple[float, float]] = self.schedule()
        if not windows:
            return [0.0, 0.0], [0.0, 0.0]

        t: List[float] = [0.0]
        y: List[float] = [0.0]
        cur = 0.0

        for s, e in windows:
            if t[-1] != s:
                t.append(s); y.append(cur)
            cur = 1.0
            t.append(s); y.append(cur)
            if t[-1] != e:
                t.append(e); y.append(cur)
            cur = 0.0
            t.append(e); y.append(cur)

        if t[-1] < windows[-1][1]:
            t.append(windows[-1][1]); y.append(0.0)
        return t, y

    def volts_at_load(self, dt_us: float, v_gen: float, r_src: float, r_load: float
                      ) -> Tuple[List[float], List[float]]:
        """
        Convert 0/1 gate to volts at the load using a simple divider model:
        V_load = V_gen * R_load / (R_src + R_load)
        """
        t, y = self.to_timeseries(dt_us=dt_us)
        denom = (r_src + r_load)
        scale = (r_load / denom) if denom > 0 else 0.0
        return t, [v * v_gen * scale for v in y]


# ----------------------------
# Sequencing many programs across time
# ----------------------------
class PulseSequence:
    """Hold many PulsePrograms at explicit start times.

    Useful when you want to chain/loop thousands of blocks without merging
    their internals. Produces a unified window schedule when needed.
    """
    def __init__(self):
        self.blocks: List[Tuple[PulseProgram, float]] = []  # (program, start_offset_us)

    def add(self, program: PulseProgram, start_offset_us: float) -> 'PulseSequence':
        self.blocks.append((program, float(start_offset_us)))
        return self

    def tile(self, program: PulseProgram, n: int, spacing_us: float) -> 'PulseSequence':
        """Append n copies of 'program' spaced by 'spacing_us' between t0 of copies.
        spacing_us can be >= program.duration_us() to create gaps.
        """
        base = 0.0
        for _ in range(n):
            self.add(program.shifted(base), base)
            base += spacing_us
        return self

    def schedule(self) -> List[Tuple[float, float]]:
        windows: List[Tuple[float, float]] = []
        for prog, off in self.blocks:
            for (s, e) in prog.schedule():
                windows.append((s + off, e + off))
        windows.sort(key=lambda w: w[0])
        # coalesce adjacent/overlapping windows
        merged: List[Tuple[float, float]] = []
        for s, e in windows:
            if not merged or s > merged[-1][1]:
                merged.append((s, e))
            else:
                ps, pe = merged[-1]
                merged[-1] = (ps, max(pe, e))
        return merged

    def duration_us(self) -> float:
        if not self.blocks:
            return 0.0
        end_times = [off + prog.duration_us() for prog, off in self.blocks]
        return max(end_times)

    def stats(self) -> Dict[str, float]:
        windows = self.schedule()
        total_on = sum((e - s) for s, e in windows)
        total = self.duration_us()
        return {
            "total_duration_us": total,
            "total_on_us": total_on,
            "avg_duty": (total_on / total) if total > 0 else 0.0,
            "num_windows": len(windows),
        }

# ----------------------------
# Convenience constructor for charge + maintain
# ----------------------------
class ThrusterPulse:
    """Two-part PWM helper (still fully general via .program).

    Typical usage:
        # 1) charge the inductor with a longer pulse
        # 2) maintain with short pulses at some PRF
    """
    def __init__(self,
                 charge_width_us: float,
                 maintain_width_us: float,
                 maintain_period_us: float,
                 maintain_count: int,
                 t0_us: float = 0.0) -> None:
        self.program = PulseProgram(t0_us=t0_us)
        # Part 1: single long pulse; period can equal width to leave no gap
        self.program.add(PulseSegment(width_us=charge_width_us,
                                      period_us=charge_width_us,
                                      count=1))
        # Part 2: short PWM train
        self.program.add(PulseSegment(width_us=maintain_width_us,
                                      period_us=maintain_period_us,
                                      count=maintain_count))

    def schedule(self) -> List[Tuple[float, float]]:
        return self.program.schedule()

    def stats(self) -> Dict[str, float]:
        return self.program.stats()

    def to_json(self) -> str:
        return self.program.to_json()


# ----------------------------
# IPM-specific convenience layers
# ----------------------------
class IPMInputPulse:
    """A single IPM gate input constructed from (charge + PWM maintain).

    This is a thin alias focused on naming for IPM use. It internally
    uses PulseProgram and PulseSegment and can be converted to a program
    or chained with others.
    """
    def __init__(self,
                 charge_width_us: float,
                 pwm_width_us: float,
                 pwm_period_us: float,
                 pwm_count: int,
                 t0_us: float = 0.0) -> None:
        self.charge_width_us = float(charge_width_us)
        self.pwm_width_us = float(pwm_width_us)
        self.pwm_period_us = float(pwm_period_us)
        self.pwm_count = int(pwm_count)
        self.t0_us = float(t0_us)

    def to_program(self) -> PulseProgram:
        prog = PulseProgram(t0_us=self.t0_us)
        prog.add(PulseSegment(self.charge_width_us, self.charge_width_us, 1))
        prog.add(PulseSegment(self.pwm_width_us, self.pwm_period_us, self.pwm_count))
        return prog

    def duration_us(self) -> float:
        return self.to_program().duration_us()

    def then(self, other: 'IPMInputPulse', gap_us: float = 0.0) -> PulseProgram:
        """Return a new PulseProgram that plays *this* IPM pulse, then *other*.
        If gap_us > 0, insert a delay between t_end(self) and t0(other).
        """
        a = self.to_program()
        b = other.to_program().shifted(a.duration_us() + float(gap_us))
        return a.extended(b)


class IPMPulseTrain:
    """A container to link many IPMInputPulse objects into one long train.

    Usage:
        train = IPMPulseTrain(spacing_us=2000.0)
        train.add(IPMInputPulse(85, 6.7, 20, 10))
        train.add(IPMInputPulse(90, 7.0, 20, 12))
        program = train.to_program()  # contiguous with spacing_us gap between blocks
    """
    def __init__(self, spacing_us: float = 0.0):
        self.spacing_us = float(spacing_us)
        self.blocks: List[IPMInputPulse] = []

    def add(self, ipm_pulse: IPMInputPulse) -> 'IPMPulseTrain':
        self.blocks.append(ipm_pulse)
        return self

    def to_program(self) -> PulseProgram:
        if not self.blocks:
            return PulseProgram()
        base = self.blocks[0].to_program()
        t = base.duration_us() + self.spacing_us
        for blk in self.blocks[1:]:
            base = base.extended(blk.to_program().shifted(t - base.t0_us))
            t += blk.duration_us() + self.spacing_us
        return base

    def to_sequence(self) -> 'PulseSequence':
        seq = PulseSequence()
        t = 0.0
        for blk in self.blocks:
            seq.add(blk.to_program().shifted(t), start_offset_us=t)
            t += blk.duration_us() + self.spacing_us
        return seq
