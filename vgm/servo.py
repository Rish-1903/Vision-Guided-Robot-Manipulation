"""Position-based visual servoing (look-then-move) for the placement step.

The camera is fixed overhead, so the gripper hides a held block. The loop is
therefore iterative look-then-move (Hutchinson, Hager and Corke, 1996):
place -> move the arm out of view -> measure the block -> regrasp and correct.

When the target is itself observed by the camera (a placement pad), the error
is measured between two image-derived poses at nearly the same location, so a
constant camera-extrinsic bias cancels. With a target given only in robot
coordinates, the loop removes actuation/grasp errors but not calibration bias.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import norm_yaw_pi, yaw_of


@dataclass
class PlanarError:
    dx: float
    dy: float
    dyaw: float

    @property
    def xy_mm(self) -> float:
        return 1000.0 * math.hypot(self.dx, self.dy)

    @property
    def yaw_deg(self) -> float:
        return abs(math.degrees(self.dyaw))


def planar_error(est_B, target_xyz, target_yaw=None) -> PlanarError:
    """Target minus measurement, in the base XY plane (yaw modulo pi)."""
    d = np.asarray(target_xyz[:2], float) - est_B[:2, 3]
    dyaw = 0.0 if target_yaw is None else norm_yaw_pi(target_yaw - yaw_of(est_B))
    return PlanarError(float(d[0]), float(d[1]), float(dyaw))


def converged(err: PlanarError, tol_mm: float, tol_deg: float) -> bool:
    return err.xy_mm < tol_mm and err.yaw_deg < tol_deg


def correction_goal(est_B, err: PlanarError, gain: float = 1.0):
    """Where to put the TCP after regrasping the block at its measured pose.

    Returns (grasp_center_B, grasp_yaw, place_center_B, place_yaw). The TCP grasps
    at the measured centre and is moved by ``gain * error``.
    """
    grasp_c = est_B[:3, 3].copy()
    grasp_yaw = yaw_of(est_B)
    place_c = grasp_c.copy()
    place_c[0] += gain * err.dx
    place_c[1] += gain * err.dy
    return grasp_c, grasp_yaw, place_c, grasp_yaw + gain * err.dyaw


def look_then_move(observe, regrasp_and_move, target_xyz, target_yaw=None, *, tol_mm=2.0,
                   tol_deg=1.5, gain=1.0, max_iters=4):
    """Generic loop. ``observe() -> B_from_O | None``; ``regrasp_and_move(goal)``.

    Returns a list of per-iteration measured errors (iteration 0 = open loop).
    """
    history = []
    for it in range(max_iters + 1):
        est = observe()
        if est is None:
            history.append({'iteration': it, 'status': 'not_observed'})
            break
        err = planar_error(est, target_xyz, target_yaw)
        history.append({'iteration': it, 'measured_xy_error_mm': err.xy_mm,
                        'measured_yaw_error_deg': err.yaw_deg})
        if converged(err, tol_mm, tol_deg):
            history[-1]['status'] = 'converged'
            break
        if it == max_iters:
            history[-1]['status'] = 'max_iterations'
            break
        regrasp_and_move(correction_goal(est, err, gain))
    return history
