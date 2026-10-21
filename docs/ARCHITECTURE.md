# System architecture

The workcell uses a **fixed eye-to-hand RGB-D camera** and a **7-DOF Franka Panda arm**.

- Task 2: estimate camera intrinsic matrix `K` (with a known unresolved checkerboard-method discrepancy).
- Task 3: estimate camera to robot-base transform `B_from_Ccv` by fitting known 3D landmark positions to image observations.
- Task 4: segment three known colors in HSV and measure image-space bounding boxes/centroids.
- Task 5: use depth and intrinsics to reconstruct cuboid top surfaces and estimate upright object pose. Object yaw is ambiguous by 180 degrees for the modeled symmetric shapes.
- Task 6: transform the camera-frame pose into the robot base, construct approach/grasp/lift/place waypoints and solve IK.

Core transform convention:

```text
T_base_object = T_base_camera_opencv @ T_camera_opencv_object
```

The code reads simulator ground-truth object transforms **only for evaluation**, not for pick planning. Camera extrinsic ground truth is used as a consistency gate, not to generate the target pose.

**Execution model:** robot has kinematic joints; block attachment is simulated via parenting to the TCP. This intentionally differs from physically realistic manipulation.
