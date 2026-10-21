"""Simulation-independent checks. Run: python -m unittest discover -s tests -v"""
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_pipeline import check_recovery_quality, validate_calibration  # noqa: E402
from vgm.config import load_destinations, load_start_layout, validate_layout  # noqa: E402


class PipelineContractTests(unittest.TestCase):
    def test_layout_all_three(self):
        validate_layout(load_start_layout(), load_destinations(), ['red', 'green', 'blue'])

    def test_reject_occupied_destination(self):
        start = load_start_layout()
        bad = dict(load_destinations())
        bad['red'] = start['blue'][:3]
        with self.assertRaises(ValueError):
            validate_layout(start, bad, ['red'])

    def test_single_destination_source(self):
        # run_pipeline must not carry its own copy of the destinations any more.
        src = (ROOT / 'run_pipeline.py').read_text()
        self.assertNotIn('DEST_WORLD', src)
        self.assertNotIn('START_WORLD', src)

    def test_destinations_file_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            f = pathlib.Path(temp) / 'd.json'
            f.write_text(json.dumps({'_comment': 'x', 'red': [0.1, 0.1, 0.05, 30],
                                     'green': [0.1, 0.0, 0.05], 'blue': [0.1, -0.1, 0.05]}))
            d = load_destinations(f)
            self.assertEqual(len(d['red']), 4)
            f.write_text(json.dumps({'red': [0.1, 0.1]}))
            with self.assertRaises(ValueError):
                load_destinations(f)

    def test_calibration_orthonormality(self):
        with tempfile.TemporaryDirectory() as temp:
            f = pathlib.Path(temp) / 'cal.npz'
            K = np.array([[831., 0, 480.], [0, 831., 360.], [0, 0, 1.]])
            np.savez(f, K=K, B_from_Ccv=np.eye(4))
            validate_calibration(f)
            bad = np.eye(4)
            bad[:3, :3] = 0.0
            np.savez(f, K=K, B_from_Ccv=bad)
            with self.assertRaises(RuntimeError):
                validate_calibration(f)

    def test_recovery_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            f = pathlib.Path(temp) / 'recovery_results.json'
            data = {'validation_projection_mean_rms_px': 0.053,
                    'simulator_ground_truth_validation_only': {
                        'translation_error_mm': 0.665, 'rotation_error_deg': 0.0401}}
            f.write_text(json.dumps(data))
            check_recovery_quality(f)
            data['simulator_ground_truth_validation_only']['translation_error_mm'] = 2250
            f.write_text(json.dumps(data))
            with self.assertRaises(RuntimeError):
                check_recovery_quality(f)


if __name__ == '__main__':
    unittest.main()
