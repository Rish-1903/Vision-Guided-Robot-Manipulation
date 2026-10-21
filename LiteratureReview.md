# Task 1 — Literature review and technical selection

**Application:** Vision-guided pick-and-place of rigid, colored cuboidal objects with a seven-joint Franka manipulator in CoppeliaSim.

**Engineering selection:** A fixed eye-to-hand perspective camera (CoppeliaSim `/Task2Camera`); Python and OpenCV; image-based planar checkerboard calibration (Zhang, 2000); known 3D colored fiducial landmarks and PnP for camera-to-base calibration; HSV segmentation and contours for 2D localization; RGB-D back-projection and a tabletop cuboid-geometry fit for constrained 6D pose estimation; SE(3) coordinate transforms and CoppeliaSim simIK for manipulator joint targets. The initial grasp demonstration uses **kinematic reparenting, not actual contact-based grasping**.

**Alternatives discussed:** PyBullet versus CoppeliaSim; ArUco+PnP for marker-based pose estimation; feature-based/machine-learning detection; IBVS/PBVS closed-loop tracking for bonus work. Classical geometry is interpretable, needs no neural-network training, and runs on the user's MacBook. Limitations: color sensitivity to illumination/background, planar tabletop and cuboid/symmetry assumptions, depth validity, TCP mounting uncertainty, collision-free path planning not guaranteed.

**Camera and calibration policy:** Estimate intrinsics using camera images (Task 2) and estimate the camera-to-base pose using observed landmarks (Task 3). Refit/verify the calibration independently because the original checkerboard-only run gave an incorrect focal length. Simulation truth may be used **only for validation**, not as the source for the camera matrix or object pick coordinates.

**Evaluation plan:** Intrinsic reprojection RMS, held-out pixel reprojection errors, camera-to-base translation and rotation errors, 2D detection rate/centroid error/IoU, symmetry-aware pose translation and rotation errors, success/failure counts for IK and executed placements, and final placement error. Save failure logs as well as successful runs. Follow up with robustness tests under occlusion, illumination shifts and different layouts.

**Primary references:**
- Z. Zhang, “A Flexible New Technique for Camera Calibration,” IEEE TPAMI, 2000. https://doi.org/10.1109/34.888718
- S. Garrido-Jurado et al., “Automatic generation and detection of highly reliable fiducial markers under occlusion,” Pattern Recognition, 2014. https://doi.org/10.1016/j.patcog.2014.01.005
- F. Chaumette and S. Hutchinson, “Visual Servo Control, Part I: Basic Approaches,” IEEE Robotics & Automation Magazine, 2006. https://doi.org/10.1109/MRA.2006.250573
- B. Wen et al., “FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects,” CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.html

This document defines Task 1's design and literature rationale; it is not a measured experiment.
