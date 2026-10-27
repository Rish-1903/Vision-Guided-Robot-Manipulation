"""Workcell configuration: one source of truth for layouts and destinations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'config'
DEFAULT_DESTINATIONS = CONFIG_DIR / 'destinations.json'
DEFAULT_START_LAYOUT = CONFIG_DIR / 'start_layout.json'
COLOR_NAMES = ('red', 'green', 'blue')

# Tabletop limits (world metres) used by layout validation.
TABLE_HALF_X = 0.32
TABLE_HALF_Y = 0.26


def _read_color_map(path: Path, lengths: tuple[int, ...], what: str) -> dict:
    obj = json.loads(Path(path).read_text())
    out = {}
    for color, coords in obj.items():
        if color.startswith('_'):
            continue  # allow "_comment" keys
        if color not in COLOR_NAMES or len(coords) not in lengths:
            raise ValueError(f'{path}: expected {what}, got {color}: {coords}')
        out[color] = tuple(map(float, coords))
    missing = set(COLOR_NAMES) - set(out)
    if missing:
        raise ValueError(f'{path}: missing entries for {sorted(missing)}')
    return out


def load_destinations(path: Path | None = None) -> dict:
    """Colour -> world (x, y, z) or (x, y, z, yaw_deg) placement centre."""
    return _read_color_map(path or DEFAULT_DESTINATIONS, (3, 4),
                           'red/green/blue -> [x, y, z] or [x, y, z, yaw_deg]')


def load_start_layout(path: Path | None = None) -> dict:
    """Colour -> world (x, y, z, yaw_deg) used ONLY to stage the scene."""
    return _read_color_map(path or DEFAULT_START_LAYOUT, (4,),
                           'red/green/blue -> [x, y, z, yaw_deg]')


def validate_layout(sources: dict, dests: dict, selected, min_dist: float = 0.110):
    """Reject placements that overlap a block still on the table at that step.

    This is a static clearance check, not a robot collision check.
    """
    current = {name: np.asarray(sources[name][:2], float) for name in COLOR_NAMES}
    for color in selected:
        target = np.asarray(dests[color][:2], float)
        if abs(float(target[0])) > TABLE_HALF_X or abs(float(target[1])) > TABLE_HALF_Y:
            raise ValueError(f'{color} placement lies outside workcell tabletop: {target}')
        for other, xy in current.items():
            if other == color:
                continue  # the picked block has left its source position
            d = float(np.linalg.norm(target - xy))
            if d < min_dist:
                raise ValueError(f'{color} placement is only {d:.3f} m from {other}. '
                                 'Move its goal in config/destinations.json.')
        current[color] = target


def latest_calibration() -> Path:
    """Most recent Task 3 calibration produced by this repository."""
    pointer = ROOT / 'latest_calibration.json'
    if pointer.is_file():
        try:
            path = Path(json.loads(pointer.read_text())['calibration_file'])
            if path.is_file():
                return path
        except (ValueError, KeyError, TypeError):
            pass
    candidates = sorted(
        (ROOT / 'runs').glob('*/task3/recovered/camera_to_robot_transform_recovered.npz'),
        key=lambda f: f.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        'No calibration from this pipeline was found. Run `run_pipeline.py --mode fresh` '
        'first, or pass --calibration /path/to/camera_to_robot_transform_recovered.npz.')
