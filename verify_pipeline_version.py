#!/usr/bin/env python3
"""Verify that this is the corrected pipeline, not the original integration."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
runner = root / 'run_pipeline.py'
collector = root / 'stages/task3/03_calibrate_camera_robot.py'
calibrator = root / 'stages/task2/02_calibrate_camera.py'
checks = {
    'runner calls Task 3 with --collect-only': '--collect-only' in runner.read_text(),
    'Task 3 supports --collect-only': "parser.add_argument('--collect-only'" in collector.read_text(),
    'Task 3 records observations without PnP': 'PnP intentionally deferred' in collector.read_text(),
    'Task 2 supports constrained intrinsics': 'K_constrained' in calibrator.read_text(),
    'Task 3 joint recovery script present': (root/'stages/task3/04_recover_intrinsics_and_extrinsics.py').is_file(),
}
print('Running corrected pipeline from:', runner)
for label,ok in checks.items():
    print(('PASS' if ok else 'FAIL'), label)
if not all(checks.values()):
    sys.exit('Wrong/incomplete version detected: do not rerun calibration yet.')
print('VERSION OK: task3 landmark collection is decoupled from old PnP K.')
