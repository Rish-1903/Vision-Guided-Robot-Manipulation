# Known limitations and planned improvements

## Camera calibration
- **Task 2 discrepancy:** the original checkerboard fit gave a focal length materially different from the nominal 60-degree, 960x720 camera.
  - *Ruled out (offline):* weak view geometry alone. With the original pose profile, `03_synthetic_selftest.py` recovers a known focal length within ~1 % at 0.3 px corner noise (24 views, 16 repeats; max error 0.96 %).
  - *Ruled out (API):* misuse of `sim.createTexture` options (1|2 = no interpolation, decal mode) or `sim.createVisionSensor` options (perspective, no fog, fixed background).
  - *To test in CoppeliaSim:* `04_render_checks.py` measures fx and fy from fronto-parallel views at known distances. It separates three causes: a FoV convention on the other image axis (623 px vs 831 px), a changed camera FoV, and texture stretch (fx differing from fy).
  - The `wide` capture profile (25-55 deg tilts, 0.45-0.95 m) is now the default and gives a 3-4x smaller focal-length spread in the self-test (std 0.13 % vs 0.37-0.47 % at 0.3 px noise).
- Task 3 places its landmark rig using the robot base pose. In simulation the base-frame landmark coordinates are known by construction. On hardware, mount the target on the flange and use robot FK, which makes this a hand-eye calibration.

## Robot / grasp modeling
- Franka's 7 joints use kinematic motion and scene dynamics are disabled. Objects are parented to the TCP dummy as a grasp proxy.
- No finger actuation, friction, contact force, collision response or block settling is simulated.
- The TCP location uses a provisional offset. The `--tcp-verified` flag is an operator confirmation, not a TCP calibration.

## Planning and safety
- Only discrete IK targets are checked. Joint-space interpolation is not collision-checked.
- Destination clearance is a static check, not a collision model.

## Evaluation
- Benchmark layouts are sampled inside the reachable, camera-visible table region. Layouts rejected by detection or IK are resampled and **counted** in the summary.
- Closed-loop correction with a robot-frame target cannot remove extrinsic calibration bias, because it converges to where the camera *believes* the target is. Use `--target-source pads` to test bias cancellation.
- The robustness lighting scenarios are image-space perturbations of rendered frames, not changes to scene lights.
- The largest-blob rule rejects small same-colour distractors. An identical same-colour block is ambiguous by design, and the robustness study measures it as a failure case.

## Future work
1. Run `04_render_checks.py` and fix the Task 2 root cause.
2. Integrate real Franka gripper geometry and a dynamic contact model.
3. Add collision-aware trajectories.
4. Swap HSV for a learned detector to handle clutter and same-colour ambiguity.
