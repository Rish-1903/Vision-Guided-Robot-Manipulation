# Calibration fix for fresh-scene Task 2 → Task 3 handoff

## What went wrong in the previous version

The original camera-only checkerboard fit returned a focal length very far from the ideal virtual camera geometry, even though it had a small reprojection RMS. Task 3A then required 6 robust PnP inliers **using that questionable intrinsic estimate** *before saving any of the colored landmark observations*. That created a dependency deadlock: Task 3B can jointly recover K and camera pose accurately from observed landmarks, but it received no dataset to operate on.

## What is changed

1. Task 2B now also computes a **constrained image-only calibration**, fitting a focal length while assuming square pixels, centered principal point, and zero lens distortion. Its focal length is **optimized from checkerboard corners** with an initial guess based on image resolution (NOT FOV metadata). This result is saved as `K_constrained`. The unconstrained and original pinhole models remain saved for scientific comparison.
2. Task 3A accepts a new `--collect-only` option. The unified pipeline always uses it, saving views when >=6 of 8 colored, noncoplanar landmarks are visible. It no longer requires PnP to accept those observations.
3. Task 3B then jointly estimates the focal length and the fixed camera-to-robot pose from all those image measurements and their **known target placement relative to the robot base**. Its held-out validation and simulator-truth comparison are checked by the existing quality gate.
4. Task 3A reuses an existing 8-marker target if you retry `--mode full` in the scene left over from the failed run, rather than creating duplicate colored spheres.

## Run from the CURRENT auto-created scene (fastest)

If CoppeliaSim still has `/Franka`, `/Task2Camera`, and the checkerboard/landmark target from the previous run, **leave that scene open and stopped**, and run:

```bash
python run_pipeline.py --mode full \
    --robot-base /Franka --camera /Task2Camera --colors red
```

`--mode full` does not recreate the robot or camera. It repeats Tasks 2–6 in that open scene and does not move robot joints unless `--execute` is specified.

## Run from a new EMPTY scene instead

Choose `File → New scene` and keep the simulation stopped; then run:

```bash
python run_pipeline.py --mode fresh \
    --robot-base /Franka --camera /Task2Camera --colors red
```

The new runner will create the camera, Franka, TCP, calibration objects and blocks as before.

## If calibration still fails

Inspect `runs/<timestamp>/task2/dataset/calibration_results/calibration_summary.json` for **constrained checkerboard calibration** and `runs/<timestamp>/task3/first_frame_raw.png`. For a robust Task 2 interpretation the independently fitted checkerboard K must match the camera's physical geometry; **the joint Task 3 recovery is not a substitute for independent Task 2 validation**. If the checkerboard K is still wrong, share representative actual checkerboard frames such as `task2/dataset/images/view_000.png` and `view_015.png` so we can inspect its texture geometry, appearance and detected corner order.

As before, the generated TCP offset is provisional. Do **not** add `--execute` before visually confirming its physical position/orientation and verifying all waypoint IK checks; an IK solution is not collision checking, and attachment is a kinematic grasp proxy rather than a physical two-finger grasp.

## Testing performed

- Python bytecode compilation completed.
- Four original pipeline contract tests passed.
- The constrained checkerboard calibration was tested on synthetic pinhole checkerboard measurements and recovered f=832.15 px from a true f=831.38 px in one test.
- The joint camera-recovery optimizer was tested with 24 synthetic noncoplanar landmark views at a true f=831.4 px; recovered 831.26 px, about 0.13 mm translation error, and about 0.006° rotation error in that specific test.
- **This patched version has NOT been run against your live CoppeliaSim installation or its rendered images.**
