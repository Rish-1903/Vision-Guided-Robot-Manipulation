"""Unit tests for the shared vgm package (no simulator required)."""
import math
import pathlib
import sys
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vgm.geometry import (make_tcp_goal, norm_yaw_pi, obj_pose_matrix, perturb_transform,  # noqa: E402
                          symmetric_yaw_error_deg, T_to_pose, translation_rotation_errors, yaw_of)
from vgm.manipulation import is_success, placement_errors  # noqa: E402
from vgm.metrics import describe, rate, success_at_thresholds, wilson_interval  # noqa: E402
from vgm.perception import (BLOCK_SIZE, BLOCK_Z, PAD_SIZE, PAD_Z, detect_from_rgb,  # noqa: E402
                            estimate_pose_from_mask)
from vgm.perturb import IMAGE_SCENARIOS  # noqa: E402
from vgm.servo import look_then_move, planar_error  # noqa: E402

W, H = 960, 720
F = (W / 2) / math.tan(math.radians(30))
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
DIST = np.zeros(5)
# OpenCV camera 1 m above the base origin, looking straight down.
B_FROM_C = np.eye(4)
B_FROM_C[:3, :3] = np.diag([1.0, -1.0, -1.0])
B_FROM_C[:3, 3] = [0.0, 0.0, 1.0]


def render_top_face(center_xy, yaw_deg, top_z, size, table_z=0.032):
    """Ray-cast depth (optical-axis Z) of a box top face over a table plane."""
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d_c = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones_like(u)], -1)
    d_b = d_c @ B_FROM_C[:3, :3].T
    o = B_FROM_C[:3, 3]
    t_top = (top_z - o[2]) / d_b[..., 2]
    p = o + d_b * t_top[..., None]
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    dx, dy = p[..., 0] - center_xy[0], p[..., 1] - center_xy[1]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    inside = (np.abs(lx) < size[0] / 2) & (np.abs(ly) < size[1] / 2)
    t_table = (table_z - o[2]) / d_b[..., 2]
    depth = np.where(inside, t_top, t_table).astype(np.float32)   # d_c.z == 1, so t == Z
    return depth, (inside * 255).astype(np.uint8)


class GeometryTests(unittest.TestCase):
    def test_pose_roundtrip(self):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', [10, -20, 30], degrees=True).as_matrix()
        T[:3, 3] = [0.1, -0.2, 0.3]
        np.testing.assert_allclose(obj_pose_matrix(T_to_pose(T)), T, atol=1e-12)

    def test_yaw_symmetry(self):
        self.assertAlmostEqual(symmetric_yaw_error_deg(math.radians(89), math.radians(-89)), 2.0, places=6)
        self.assertAlmostEqual(norm_yaw_pi(math.pi), 0.0, places=9)
        self.assertAlmostEqual(norm_yaw_pi(math.radians(100)), math.radians(-80), places=9)

    def test_symmetric_rotation_error(self):
        A = np.eye(4)
        B = np.eye(4)
        B[:3, :3] = Rotation.from_euler('z', 180, degrees=True).as_matrix()
        _, raw, sym = translation_rotation_errors(A, B)
        self.assertAlmostEqual(raw, 180.0, places=6)
        self.assertAlmostEqual(sym, 0.0, places=6)

    def test_tcp_goal_points_down(self):
        T = make_tcp_goal([0.3, 0.1, 0.05], math.radians(30), 0.09)
        np.testing.assert_allclose(T[:3, 2], [0, 0, -1], atol=1e-12)
        self.assertAlmostEqual(T[2, 3], 0.14)
        self.assertAlmostEqual(np.linalg.det(T[:3, :3]), 1.0)

    def test_perturb_identity(self):
        np.testing.assert_allclose(perturb_transform(B_FROM_C), B_FROM_C)


class MetricTests(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson_interval(20, 20)
        self.assertEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.8389, places=3)   # 20/20 is NOT "100 % reliable"
        self.assertEqual(rate(0, 0)['rate'], None)

    def test_describe_and_thresholds(self):
        d = describe([1.0, 2.0, None, float('nan'), 3.0])
        self.assertEqual(d['n'], 3)
        self.assertAlmostEqual(d['mean'], 2.0)
        s = success_at_thresholds([1.0, 4.0, None, 12.0], thresholds_mm=(5, 10))
        self.assertEqual(s['<5mm']['successes'], 2)
        self.assertEqual(s['<5mm']['n'], 4)

    def test_success_criterion(self):
        T = np.eye(4)
        T[:3, 3] = [0.104, 0.0, 0.052]
        err = placement_errors(T, [0.1, 0.0, 0.052])
        self.assertTrue(is_success(err))
        T[:3, 3] = [0.115, 0.0, 0.052]                # 15 mm off: would pass the old 30 mm gate
        self.assertFalse(is_success(placement_errors(T, [0.1, 0.0, 0.052])))


class PerceptionTests(unittest.TestCase):
    def test_pose_from_synthetic_rgbd(self):
        for xy, yaw in [((0.10, -0.05), 25.0), ((-0.18, 0.12), -70.0), ((0.0, 0.0), 0.0)]:
            top = BLOCK_Z + BLOCK_SIZE[2] / 2
            depth, mask = render_top_face(xy, yaw, top, BLOCK_SIZE)
            T, info = estimate_pose_from_mask(depth, mask, K, DIST, B_FROM_C)
            self.assertIsNotNone(T, info)
            self.assertLess(1000 * np.linalg.norm(T[:2, 3] - np.asarray(xy)), 1.0)
            self.assertLess(1000 * abs(T[2, 3] - BLOCK_Z), 0.5)
            self.assertLess(symmetric_yaw_error_deg(yaw_of(T), math.radians(yaw)), 1.0)

    def test_range_depth_mode(self):
        top = BLOCK_Z + BLOCK_SIZE[2] / 2
        depth, mask = render_top_face((0.2, 0.1), 10.0, top, BLOCK_SIZE)
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        ray = np.sqrt(1 + ((u - K[0, 2]) / F) ** 2 + ((v - K[1, 2]) / F) ** 2)
        T, _ = estimate_pose_from_mask((depth * ray).astype(np.float32), mask, K, DIST, B_FROM_C,
                                       depth_mode='range')
        self.assertLess(1000 * np.linalg.norm(T[:2, 3] - [0.2, 0.1]), 1.0)

    def test_largest_blob_rejects_distractor(self):
        img = np.full((H, W, 3), 60, np.uint8)
        cv2.rectangle(img, (300, 300), (420, 380), (20, 20, 240), -1)     # block
        cv2.rectangle(img, (700, 100), (725, 125), (20, 20, 240), -1)     # small red distractor
        det, masks = detect_from_rgb(img)
        self.assertEqual(det['red']['n_blobs'], 2)
        self.assertLess(abs(det['red']['centroid'][0] - 360), 1.5)
        self.assertEqual(int(masks['red'][110, 710]), 0)
        self.assertIsNone(det['green'])

    def test_image_scenarios_shapes(self):
        rng = np.random.default_rng(0)
        img = np.full((40, 50, 3), 128, np.uint8)
        depth = np.ones((40, 50), np.float32)
        for name, (f_rgb, f_depth) in IMAGE_SCENARIOS.items():
            if f_rgb:
                self.assertEqual(f_rgb(img, rng).shape, img.shape, name)
            if f_depth:
                self.assertEqual(f_depth(depth, rng).shape, depth.shape, name)


class ServoTests(unittest.TestCase):
    """Closed-loop look-then-move under an injected camera calibration error.

    True block motion is rigid with the TCP, so placing moves the block by
    (commanded place - commanded grasp). Measurements go through the WRONG
    extrinsics. With a camera-observed pad as the target, the bias common to
    block and pad cancels (up to the 36 mm height difference between the two
    measured surfaces); with a robot-frame target it cannot.
    """

    def _run(self, target_source, rot_deg, closed):
        B_C_est = perturb_transform(B_FROM_C, rot_deg, (0, 0, 0))
        E = B_C_est @ np.linalg.inv(B_FROM_C)       # true base point -> measured base point
        block_top_h = BLOCK_Z + BLOCK_SIZE[2] / 2
        pad_true = np.array([0.25, 0.10, PAD_Z + PAD_SIZE[2] / 2])
        state = {'xy': np.array([-0.15, -0.10])}

        def measure(p):
            return (E @ np.r_[p, 1.0])[:3]

        def observe():
            m = measure(np.r_[state['xy'], block_top_h])
            T = np.eye(4)
            T[:3, 3] = m - [0, 0, BLOCK_SIZE[2] / 2]
            return T

        def move(goal):
            grasp_c, _, place_c, _ = goal
            state['xy'] = state['xy'] + (place_c[:2] - grasp_c[:2])

        if target_source == 'pads':
            target = measure(pad_true)[:3]
        else:
            target = pad_true.copy()
        est = observe()
        move((est[:3, 3], 0.0, target, 0.0))            # open-loop placement
        if closed:
            look_then_move(observe, move, target, None, tol_mm=0.5, tol_deg=90, max_iters=5)
        return 1000 * np.linalg.norm(state['xy'] - pad_true[:2])

    def test_open_loop_suffers_from_bias(self):
        self.assertGreater(self._run('world', (1.0, 0.0, 0.0), closed=False), 10.0)

    def test_closed_loop_on_world_target_cannot_remove_bias(self):
        self.assertGreater(self._run('world', (1.0, 0.0, 0.0), closed=True), 10.0)

    def test_closed_loop_on_observed_pad_cancels_bias(self):
        for rot in [(1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0), (0.7, -0.7, 0.5)]:
            self.assertLess(self._run('pads', rot, closed=True), 1.5, rot)

    def test_no_bias_open_loop_exact(self):
        self.assertLess(self._run('world', (0, 0, 0), closed=False), 1e-6)

    def test_planar_error_yaw_wrap(self):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('z', 179, degrees=True).as_matrix()
        e = planar_error(T, [0, 0, 0], target_yaw=0.0)
        self.assertAlmostEqual(e.yaw_deg, 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
