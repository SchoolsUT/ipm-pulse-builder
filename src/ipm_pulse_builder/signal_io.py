#!/usr/bin/env python3
"""
signal_io.py

Save/load editable IPM pulse "projects" (not CSV exports).
Stores:
  - Program (pre_gap_us + ordered blocks + all block parameters)
  - Optional SDG settings dict (host/channel/npoints/high_v/low_v/load/trigger_source/etc.)
  - Optional UI state dict

File is JSON, versioned, and includes a migration hook.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple, Union, Optional

from pulse_schedule import Program, Gap, ChargePWM

PROJECT_FORMAT = "ipm-pulse-builder"
PROJECT_VERSION = 1


# ----------------------------
# Helpers (safe casting)
# ----------------------------

def _as_float(x: Any, field: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Invalid float for '{field}': {x!r}") from e


def _as_int(x: Any, field: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"Invalid int for '{field}': {x!r}") from e


def _as_str(x: Any, field: str) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # allow numbers/bools to stringify, but keep it explicit
    return str(x)


def _expect_dict(x: Any, field: str) -> Dict[str, Any]:
    if x is None:
        return {}
    if not isinstance(x, dict):
        raise ValueError(f"Expected object for '{field}', got {type(x).__name__}")
    return x


def _expect_list(x: Any, field: str) -> list:
    if x is None:
        return []
    if not isinstance(x, list):
        raise ValueError(f"Expected array for '{field}', got {type(x).__name__}")
    return x


# ----------------------------
# Block <-> dict
# ----------------------------

BlockT = Union[Gap, ChargePWM]

def block_to_dict(b: BlockT) -> Dict[str, Any]:
    if isinstance(b, Gap):
        return {"type": "Gap", "gap_us": float(b.gap_us)}
    if isinstance(b, ChargePWM):
        return {
            "type": "ChargePWM",
            "charge_width_us": float(b.charge_width_us),
            "pwm_width_us": float(b.pwm_width_us),
            "pwm_period_us": float(b.pwm_period_us),
            "pwm_count": int(b.pwm_count),
        }
    raise TypeError(f"Unsupported block type: {type(b)}")


def block_from_dict(d: Dict[str, Any]) -> BlockT:
    t = _as_str(d.get("type", ""), "items[].type")
    if t == "Gap":
        return Gap(gap_us=_as_float(d.get("gap_us", 0.0), "items[].gap_us"))
    if t == "ChargePWM":
        return ChargePWM(
            charge_width_us=_as_float(d.get("charge_width_us", 0.0), "items[].charge_width_us"),
            pwm_width_us=_as_float(d.get("pwm_width_us", 0.0), "items[].pwm_width_us"),
            pwm_period_us=_as_float(d.get("pwm_period_us", 0.0), "items[].pwm_period_us"),
            pwm_count=_as_int(d.get("pwm_count", 0), "items[].pwm_count"),
        )
    raise ValueError(f"Unknown block type: {t!r}")


# ----------------------------
# Program <-> dict
# ----------------------------

def program_to_dict(p: Program) -> Dict[str, Any]:
    return {
        "pre_gap_us": float(getattr(p, "pre_gap_us", 0.0)),
        "items": [block_to_dict(it) for it in getattr(p, "items", [])],
    }


def program_from_dict(d: Dict[str, Any]) -> Program:
    dd = _expect_dict(d, "program")
    p = Program()
    p.pre_gap_us = _as_float(dd.get("pre_gap_us", 0.0), "program.pre_gap_us")
    items = _expect_list(dd.get("items", []), "program.items")
    for it in items:
        it_d = _expect_dict(it, "program.items[]")
        p.add(block_from_dict(it_d))
    return p


# ----------------------------
# Migration hook
# ----------------------------

def _migrate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate older versions to PROJECT_VERSION.
    Keep this strict: only accept known versions.
    """
    v = int(payload.get("version", 0))
    if v == PROJECT_VERSION:
        return payload

    # Example future migration pattern:
    # if v == 0:
    #     ... mutate payload ...
    #     payload["version"] = 1
    #     return payload

    raise ValueError(f"Unsupported project version: {v} (expected {PROJECT_VERSION})")


# ----------------------------
# Public API
# ----------------------------

def save_project(
    path: str,
    program: Program,
    *,
    sdg: Optional[Dict[str, Any]] = None,
    ui: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "program": program_to_dict(program),
        "sdg": sdg or {},
        "ui": ui or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_project(path: str) -> Tuple[Program, Dict[str, Any], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    payload = _expect_dict(payload, "root")
    fmt = payload.get("format", "")
    if fmt != PROJECT_FORMAT:
        raise ValueError(f"Not an {PROJECT_FORMAT!r} project file (format={fmt!r}).")

    payload = _migrate_payload(payload)

    program = program_from_dict(payload.get("program", {}))
    sdg = _expect_dict(payload.get("sdg", {}), "sdg")
    ui = _expect_dict(payload.get("ui", {}), "ui")
    return program, sdg, ui

def load_signal_project(path: str) -> Tuple[Program, Dict[str, Any], Dict[str, Any]]:
    """
    Load an editable IPM project from disk.

    CSV files are waveform exports and cannot be rebuilt into editable pulse blocks.
    This only accepts saved IPM project files handled by load_project(...).

    Returns:
        (program, sdg_settings, ui_state)
    """
    if str(path).lower().endswith(".csv"):
        raise ValueError(
            "CSV files are waveform exports and are not editable IPM project files. "
            "Load a saved .ipm project file instead."
        )
    return load_project(path)
