"""Simulation-independent checks. Run: python -m unittest discover -s tests -v"""
import json
import pathlib
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from run_pipeline import (validate_layout, START_WORLD, DEST_WORLD,
                          validate_calibration, check_recovery_quality,
                          load_destinations)

class PipelineContractTests(unittest.TestCase):
    def test_layout_all_three(self):
        validate_layout(START_WORLD,DEST_WORLD,['red','green','blue'])

    def test_reject_occupied_destination(self):
        bad = dict(DEST_WORLD)
        bad['red'] = START_WORLD['blue'][:3]
        with self.assertRaises(ValueError):
            validate_layout(START_WORLD,bad,['red'])

    def test_calibration_orthonormality(self):
        with tempfile.TemporaryDirectory() as temp:
            f=pathlib.Path(temp)/'cal.npz'
            K=np.array([[831.,0,480.],[0,831.,360.],[0,0,1.]])
            np.savez(f,K=K,B_from_Ccv=np.eye(4))
            validate_calibration(f)
            bad = np.eye(4)
            bad[:3,:3] = 0.0
            np.savez(f,K=K,B_from_Ccv=bad)
            with self.assertRaises(RuntimeError):
                validate_calibration(f)

    def test_recovery_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            f=pathlib.Path(temp)/'recovery_results.json'
            data={'validation_projection_mean_rms_px': 0.053,
                  'simulator_ground_truth_validation_only': {
                      'translation_error_mm':0.665,
                      'rotation_error_deg':0.0401}}
            f.write_text(json.dumps(data))
            check_recovery_quality(f)
            data['simulator_ground_truth_validation_only']['translation_error_mm']=2250
            f.write_text(json.dumps(data))
            with self.assertRaises(RuntimeError):
                check_recovery_quality(f)

if __name__ == '__main__':
    unittest.main()
