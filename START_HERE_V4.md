# V4 — Entire scene kinematic (NO physics) for safe visual manipulation demonstration

## What changed from V3
V3 switched the Franka's seven joints to kinematic mode but left scene dynamics
running. Dynamic robot links could fall apart despite joint angles not changing.
V4 disables **scene dynamics BEFORE changing joint modes**. It verifies this setting,
then checks both joint drift AND every Franka shape's pose during an idle test.
This fixes a real consistency issue in the previous package, but live compatibility
still requires verification on your Mac and exact CoppeliaSim model.

**This means there is NO gravity, physical collision/contact, force/torque or
finger grasp simulation.** The robot and the transported blocks move kinematically;
the block is parented to the TCP for the simulated grasp. Do not describe it as a
physics-based physical grasp in a report. Motion paths are not collision-checked.

## How to recover
1. Stop simulation. **Discard the V3 scene without saving.** If it was saved, do not
   use it; load a clean, newly imported Franka in a blank scene.
2. Unzip V4 to its own folder. Activate your installed Python environment.
3. Open CoppeliaSim -> File -> New scene (simulation stopped).
4. Run the **fresh mode dry run**:

```bash
python run_pipeline.py --mode fresh --robot-base /Franka --camera /Task2Camera --colors red green blue
```

Expected early log: `KINEMATIC-ONLY DEMO: entire scene physics disabled ... verified OFF.`
Expected check: `IDLE CHECK ... link translation drift ...` and `IDLE CHECK PASSED`.
If the robot still separates, STOP, share a screenshot of its scene hierarchy
and the idle-check output; do not execute anything.

The fresh run should save `runs/<timestamp>/workcell_ready_for_runtime.ttt`.
Open that scene and verify the provisional TCP at the intended grasp center;
the original arbitrary 10cm offset may be wrong. Save after adjusting.

## Manipulate all 3, sequentially
Only AFTER you visually check that the arm remains assembled and the TCP is
correct, test the runtime **without** `--execute` first:

```bash
python run_pipeline.py --mode runtime --robot-base /Franka --camera /Task2Camera --colors red green blue --tcp-verified
```

Then execute:

```bash
python run_pipeline.py --mode runtime --robot-base /Franka --camera /Task2Camera --colors red green blue --tcp-verified --execute
```

The code has separate destinations in `config/destinations.json`, preflights the
six IK waypoints for all three blocks before motion, then performs
red -> green -> blue with fresh RGB-D pose estimation before each pick.
An IK solution does not prove collision-free motion. Be ready to stop simulation.

NOTE: The individual checkerboard Task 2 intrinsic estimate is still inconsistent
with a 60-degree, 960x720 camera and needs independent study before submission.

## Testing status
Syntax & mock unit tests: passed in the development container.
Live CoppeliaSim simulation on your Mac: **NOT verified**.
