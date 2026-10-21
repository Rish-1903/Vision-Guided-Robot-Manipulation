# Known limitations and planned improvements

## Camera calibration
- **Task 2 discrepancy:** the independent checkerboard fit gave a focal length materially different from the nominal 60-degree, 960x720 camera model. Task 3's joint 3D-2D landmark fitting yielded a good calibration for downstream processing, but the original Task 2 discrepancy remains an open issue.
- Recommended next step: audit checkerboard square spacing, number/order of inner corners, RGB frame orientation, image aspect ratio and 3D board geometry. Refit from diverse checkerboard poses and compare to simulator truth for **validation only**.

## Robot / grasp modeling
- Franka's 7 joints use kinematic motion and scene dynamics are disabled for this demonstration. Objects are parented to the tool-center dummy as a grasping proxy.
- No physical gripper contact, finger actuation, friction, contact force, collision response or block settling is simulated.
- TCP location is configured with a provisional offset. Verify this against the actual intended tool geometry; do not equate the `--tcp-verified` flag with a measured TCP calibration.

## Planning and safety
- Only discrete IK targets are checked. Joint-space interpolation can encounter obstacles even when waypoint IK solves.
- A single randomized IK attempt can fail on a reachable pose. Multiple seeds, continuity constraints, joint-limit costs and collision-aware planning are future work.
- Proposed drop-off targets must be checked for collisions with other blocks and the robot. The static minimum-separation check is not a full collision model.

## Evaluation
- Report localization and pose accuracy separately from manipulation success; they are different measurements.
- Repeat pick-and-place across many independent initial object layouts, with success criterion, placement tolerance, repetitions, and failure reasons logged.
- Do not claim generalization to uncontrolled lighting, occlusion, novel shapes or physical hardware unless evaluated.

## Future work
1. Correct checkerboard-based intrinsic calibration independently.
2. Integrate actual Franka gripper geometry and a dynamic contact model.
3. Add collision-aware Cartesian/joint trajectories and smooth continuity planning.
4. Test generalization and benchmark multi-trial pick-place success.
