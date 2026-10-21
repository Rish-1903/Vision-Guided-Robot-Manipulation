# System architecture

The workcell uses a **fixed eye-to-hand RGB-D camera** and a **7-DOF Franka Panda arm**.

- Task 2: estimate the camera intrinsic matrix `K` from checkerboard views. `03_synthetic_selftest.py` and `04_render_checks.py` diagnose the known checkerboard discrepancy.
- Task 3: estimate the camera-to-robot-base transform `B_from_Ccv` by fitting known 3D landmark positions to image observations.
- Task 4: segment three known colours in HSV. The largest blob per colour is kept and the number of candidate blobs is reported.
- Task 5: back-project masked depth, fit the cuboid top face (SVD plane plus minimum-area rectangle) and offset by half the block height. Yaw is ambiguous by 180 degrees.
- Task 6: build approach/grasp/lift/place waypoints in the base frame, solve them with simIK, interpolate in joint space and attach the block kinematically.
- Bonus: look-then-move PBVS (`vgm/servo.py`). After placement the arm moves clear, the camera re-measures the block and the robot regrasps and corrects.

## Code layout

```text
vgm/geometry.py      pose conversions, symmetric yaw/rotation errors, TCP goals, error injection
vgm/config.py        destinations / start layout (config/*.json), layout validation, calibration lookup
vgm/perception.py    HSV detection, projection, RGB-D cuboid and pad pose estimation
vgm/scene.py         CoppeliaSim I/O: images, depth, poses, block/pad creation, random layouts
vgm/manipulation.py  TCP discovery, IKMotion, live perception, placement scoring
vgm/servo.py         planar error, correction goal, look_then_move loop
vgm/perturb.py       image-space lighting / noise perturbations
vgm/metrics.py       describe(), Wilson intervals, success-at-threshold
```

Every stage script imports from `vgm`. There are no duplicated helper modules.

## Transform convention

```text
T_base_object = T_base_camera_opencv @ T_camera_opencv_object
```

## Ground-truth policy

Simulator object and pad poses are read **only after** inference or motion, to score results. The simulator camera pose is used only as a consistency gate in `run_pipeline.py` (rejecting a calibration that belongs to a different scene) and in the Task 2 render diagnostic. It never generates a target.

**Execution model:** kinematic joints and physics disabled. Block attachment is simulated by parenting the block to the TCP.
