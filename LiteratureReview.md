# Task 1: Literature review and technical selection

**Application.** Vision-guided pick-and-place of rigid, coloured cuboids (82 x 60 x 38 mm) by a 7-DOF Franka Emika Panda in CoppeliaSim, observed by one fixed overhead RGB-D camera (eye-to-hand, 960 x 720, 60 deg FoV).

For each stage we summarise the principle, the alternatives, why we chose what we chose, how it plugs into the next stage, and where it breaks. Selection criteria throughout: (1) correctness we can verify against simulator ground truth, (2) no training data needed, (3) interpretable failure modes, (4) runs on a laptop CPU.

## 1. Camera intrinsic calibration (Task 2)

**Principle.** Zhang's method [1]: each view of a planar target gives a homography H = K [r1 r2 t]. Orthonormality of r1, r2 gives two linear constraints on the image of the absolute conic B = K^-T K^-1, so three or more views fix K in closed form; a Levenberg-Marquardt refinement then minimises reprojection error jointly over K, distortion and every board pose.

| Option | Principle | Pros | Cons | Verdict |
|---|---|---|---|---|
| Checkerboard + Zhang [1] | planar homographies | standard, sub-pixel saddle corners, OpenCV built-in | needs well-spread, strongly tilted views | **chosen** |
| ChArUco board [2] | checkerboard + ArUco IDs | works with partial views, unique corner IDs | extra dependency, no accuracy gain on clean renders | good alternative |
| Read simulator FoV | f = (w/2)/tan(FoV/2) | exact in simulation | not a calibration; does not transfer to hardware | **validation only** |

**Integration.** K feeds PnP in Task 3 and back-projection in Task 5.
**Limitations.** Zhang's method is ill-conditioned for near-fronto-parallel views: focal length and board distance trade off. Our synthetic self-test (`stages/task2/03_synthetic_selftest.py`) quantifies this: with the original capture profile (tilts <= 27 deg, board ~0.9 m away) the focal-length standard deviation is 0.37-0.47 % at 0.3 px corner noise (worst case 0.96 %), versus 0.12-0.13 % with 25-55 deg tilts and 0.45-0.95 m depth. Since even the weak profile gives <1 % error, the larger Task 2 mismatch must come from the rendering setup, which `04_render_checks.py` tests directly.

## 2. Camera-to-robot (eye-to-hand) calibration (Task 3)

**Principle.** Find B_T_C, the camera pose in the robot base frame. Two families exist:
- *Hand-eye (AX = XB)* [3, 4]: move a target mounted on the gripper through N robot poses, observe it each time, and solve for the unknown transform from pairs of relative motions.
- *Direct 3D-2D (PnP)* [5]: if 3D landmark positions are known in the base frame, solve PnP for the camera pose, optionally jointly with K.

| Option | Needs | Pros | Cons | Verdict |
|---|---|---|---|---|
| Tsai-Lenz / Park-Martin AX=XB [3, 4] | target on gripper, robot FK | standard on real robots, no base-frame landmark coordinates needed | needs rotation-rich motions; noise amplification on small rotations | best for hardware |
| Landmark PnP / joint K+pose fit (EPnP init + LM) [5] | landmarks with known base-frame coordinates | very accurate in simulation, few views | landmark coordinates must come from somewhere | **chosen** |
| ArUco markers + PnP [2] | printed markers | robust detection, IDs | marker pose noise grows with distance | alternative |

**Integration.** B_T_C converts every Task 5 pose into the frame used by IK.
**Limitations.** Our landmark rig is placed relative to the robot base. In simulation, its base-frame coordinates are known by construction. On hardware they would come from robot FK (target held by the gripper), which turns the method into a hand-eye calibration. We report this honestly. Held-out landmark reprojection (0.053 px) measures fit consistency; the comparison with simulator camera pose (0.665 mm) is validation only.

## 3. Object localization (Task 4)

| Option | Principle | Pros | Cons | Verdict |
|---|---|---|---|---|
| HSV threshold + contours [6] | colour segmentation, border following | zero training, real-time, transparent | lighting- and colour-dependent; one blob per colour | **chosen** |
| Fiducial markers (ArUco) [2] | coded squares, PnP | gives ID and 6D pose directly | requires marking objects | rejected (changes the task) |
| Learned detector (YOLO-family, Mask R-CNN) | CNN detection/segmentation | robust to clutter and lighting | needs labelled data and GPU; overkill for 3 known colours | future work |

**Integration.** Each colour mask (largest blob only, so same-colour clutter does not corrupt depth) is handed to Task 5.
**Limitations, measured.** `stages/task4/task4_robustness.py` measures the failure envelope on paired layouts: illumination gain/gamma/colour cast, RGB and depth noise, same-colour distractors, partial occlusion of 25/50/75 %, and touching blocks.

## 4. 6D pose estimation (Task 5)

| Option | Principle | Pros | Cons | Verdict |
|---|---|---|---|---|
| Constrained geometric fit (ours) | back-project masked depth, SVD plane fit of top face [7], min-area rectangle for yaw | exact model of our object, sub-mm in sim, no training | assumes upright cuboid with visible top face; yaw ambiguous by 180 deg | **chosen** |
| Model-based registration (ICP / RANSAC [8]) | align CAD model to point cloud | handles arbitrary orientations | needs a good initial guess; local minima | alternative |
| Learned 6D pose (PoseCNN [9], FoundationPose [10]) | CNN / foundation model, model-based or model-free | generalises to novel objects and clutter | GPU, heavy dependencies, harder to verify | future work |

**Integration.** B_T_O = B_T_C * C_T_O. The pose is converted into a top-down TCP goal at the block centre plus the tool offset.
**Limitations.** Symmetric errors are reported modulo 180 deg. Tilted or stacked blocks are out of scope. The fit needs at least 100 top-face depth points, so heavy occlusion fails closed and does not produce a wrong pose.

## 5. Manipulation and closed-loop correction (Task 6, bonus)

| Option | Principle | Pros | Cons | Verdict |
|---|---|---|---|---|
| Open-loop look-and-move | perceive once, plan, execute | simple | all calibration and actuation error goes straight into placement | **baseline** |
| PBVS, look-then-move [11, 12] | re-measure pose error in 3D, correct iteratively | direct Cartesian control; robust to actuation error | inherits calibration error *unless the target is also observed* | **chosen (bonus)** |
| IBVS [11] | control on image-feature error | robust to calibration error | needs a goal image and features visible during motion | not suited: gripper occludes the block from the overhead camera |

**Key design point.** Our camera is overhead and the gripper hides a held block, so continuous servoing is impossible. We servo between grasps: place, move the arm clear, re-measure, regrasp and correct. When the target is a pad that the camera also observes, block and target are measured through the *same* extrinsics, so a constant calibration bias cancels. This is the classic argument for endpoint-closed-loop control [11]. Our offline simulation in `tests/test_vgm.py` gives these numbers for an injected 1 deg camera error:

| Target and loop | 1 deg tilt (rx) | 1 deg yaw (rz) |
|---|---|---|
| Robot-frame target, open loop | 16.2 mm | 3.2 mm |
| Robot-frame target, closed loop | 16.2 mm | 4.7 mm |
| Observed pad, open loop | 0.6 mm | 7.8 mm |
| Observed pad, closed loop | **0.6 mm** | **0.1 mm** |

Closed loop on a robot-frame target cannot remove calibration bias, because it converges to where the camera *thinks* the target is. The live benchmark (`task6_benchmark.py --target-source pads --closed-loop --inject-rot-deg ...`) repeats this experiment in CoppeliaSim.

**Grasp model.** Kinematic attachment (physics disabled), which isolates perception-to-placement accuracy from contact dynamics. Real grasp physics (friction, finger closure, slip) is out of scope.

## 6. Simulator

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| CoppeliaSim [13] | Franka model, RGB-D vision sensors, simIK, ZeroMQ Python API, Edu licence | slower than PyBullet; remote-API latency | **chosen** |
| PyBullet [14] | fast, pip install, good for RL | basic rendering; IK and camera models less complete | alternative |
| Isaac Sim / MuJoCo | photorealism / fast accurate contact | GPU (Isaac); more setup | out of scope |

## 7. Evaluation methodology

Every metric is computed against simulator ground truth that is read *after* inference. Success rates come with 95 % Wilson score intervals [15], since 20/20 successes only bound the true rate above ~84 %. Placement success uses 10 mm XY, 5 mm Z and 5 deg yaw tolerances, which are tight relative to the 60 mm block width. The full error distribution and success-at-threshold curve (2-30 mm) are reported alongside, so the threshold choice is transparent.

## References

1. Z. Zhang, "A flexible new technique for camera calibration," IEEE TPAMI 22(11), 2000. https://doi.org/10.1109/34.888718
2. S. Garrido-Jurado et al., "Automatic generation and detection of highly reliable fiducial markers under occlusion," Pattern Recognition 47(6), 2014. https://doi.org/10.1016/j.patcog.2014.01.005
3. R. Y. Tsai and R. K. Lenz, "A new technique for fully autonomous and efficient 3D robotics hand/eye calibration," IEEE Trans. Robotics and Automation 5(3), 1989. https://doi.org/10.1109/70.34770
4. F. C. Park and B. J. Martin, "Robot sensor calibration: solving AX = XB on the Euclidean group," IEEE Trans. Robotics and Automation 10(5), 1994. https://doi.org/10.1109/70.326576
5. V. Lepetit, F. Moreno-Noguer and P. Fua, "EPnP: An accurate O(n) solution to the PnP problem," IJCV 81, 2009. https://doi.org/10.1007/s11263-008-0152-6
6. S. Suzuki and K. Abe, "Topological structural analysis of digitized binary images by border following," CVGIP 30(1), 1985. https://doi.org/10.1016/0734-189X(85)90016-7
7. K. S. Arun, T. S. Huang and S. D. Blostein, "Least-squares fitting of two 3-D point sets," IEEE TPAMI 9(5), 1987. https://doi.org/10.1109/TPAMI.1987.4767965
8. M. A. Fischler and R. C. Bolles, "Random sample consensus," Communications of the ACM 24(6), 1981. https://doi.org/10.1145/358669.358692
9. Y. Xiang et al., "PoseCNN: A convolutional neural network for 6D object pose estimation in cluttered scenes," RSS 2018. https://doi.org/10.15607/RSS.2018.XIV.019
10. B. Wen et al., "FoundationPose: Unified 6D pose estimation and tracking of novel objects," CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.html
11. S. Hutchinson, G. D. Hager and P. I. Corke, "A tutorial on visual servo control," IEEE Trans. Robotics and Automation 12(5), 1996. https://doi.org/10.1109/70.538972
12. F. Chaumette and S. Hutchinson, "Visual servo control, Part I: Basic approaches," IEEE RAM 13(4), 2006. https://doi.org/10.1109/MRA.2006.250573
13. E. Rohmer, S. P. N. Singh and M. Freese, "V-REP: A versatile and scalable robot simulation framework," IROS 2013. https://doi.org/10.1109/IROS.2013.6696520
14. E. Coumans and Y. Bai, "PyBullet, a Python module for physics simulation for games, robotics and machine learning," 2016-2021. https://pybullet.org
15. E. B. Wilson, "Probable inference, the law of succession, and statistical inference," JASA 22(158), 1927. https://doi.org/10.1080/01621459.1927.10502953
