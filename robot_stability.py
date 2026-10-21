#!/usr/bin/env python3
"""Kinematic-only Franka workcell setup for the *visual grasp proxy* project.

Avoids the V3 error of switching individual joints to kinematic while
leaving dynamic bodies governed by a physics solver. THIS DISABLES ALL
SCENE DYNAMICS: there are no physical contacts, falling objects or gravity.
Do not use for experiments claiming physical grasping/contact mechanics.
"""
from __future__ import annotations
import math
import time


def arm_joints(sim, base):
    joints = [h for h in sim.getObjectsInTree(base, sim.object_joint_type, 0)]
    depth_and_joint=[]
    for h in joints:
        node=h; depth=0
        while node != -1 and node != base:
            if sim.getObjectType(node)==sim.object_joint_type:
                depth+=1
            node=sim.getObjectParent(node)
        if node==base and 1 <= depth <= 7:
            depth_and_joint.append((depth,h))
    result=[]
    for depth in range(1,8):
        matches=[h for d,h in depth_and_joint if d==depth]
        if len(matches)!=1:
            raise RuntimeError(f'Cannot identify unique Franka arm joint at depth {depth}: {matches}')
        result.append(matches[0])
    return result


def disable_all_scene_dynamics(sim):
    """Turn off the physics engine BEFORE modifying individual joint modes.

    Modern API scene property preferred; legacy fallback for older CoppeliaSim.
    Fail closed instead of permitting a dynamic/kinematic mismatch.
    """
    if sim.getSimulationState()!=sim.simulation_stopped:
        raise RuntimeError('STOP simulation before disabling dynamics.')
    method='scene property'
    try:
        sim.setBoolProperty(sim.handle_scene, 'dynamicsEnabled', False)
        value=bool(sim.getBoolProperty(sim.handle_scene,'dynamicsEnabled'))
    except Exception as first_error:
        try:
            sim.setBoolParam(sim.boolparam_dynamics_handling_enabled, False)
            value=bool(sim.getBoolParam(sim.boolparam_dynamics_handling_enabled))
            method='legacy parameter'
        except Exception as old_error:
            raise RuntimeError('Cannot disable scene dynamics; refusing to alter robot joints. '
                               f'property error={first_error}; legacy error={old_error}') from old_error
    if value:
        raise RuntimeError('Scene dynamics still ENABLED. Refusing to switch individual Franka joints; '
                           'that mismatch can make the robot links separate.')
    print(f'KINEMATIC-ONLY DEMO: entire scene physics disabled ({method}), verified OFF.')
    print('This means gripper contact and falling/settling blocks are NOT physics-simulated.')


def hold_arm_kinematically(sim, base, *, log=True):
    if sim.getSimulationState()!=sim.simulation_stopped:
        raise RuntimeError('STOP simulation before configuring hold mode.')
    # Important: turn off physics globally; a dynamically modeled Franka may
    # fall apart if only its joint modes are changed (V3 regression).
    disable_all_scene_dynamics(sim)
    try:
        sim.setBoolProperty(base, 'model.scriptsNotActive', True)
        if log:
            print('ROBOT IDLE HOLD: disabled embedded Franka model scripts.')
    except Exception as exc:
        raise RuntimeError('Cannot disable embedded Franka scripts; refusing unverified kinematic setup.') from exc
    joints=arm_joints(sim,base)
    original_positions=[float(sim.getJointPosition(j)) for j in joints]
    for j,q in zip(joints,original_positions):
        sim.setJointMode(j,sim.jointmode_kinematic)
        sim.setJointPosition(j,q)
        sim.setJointTargetPosition(j,q)
    if log:
        print('ROBOT IDLE HOLD: 7 kinematic joints locked at', [round(q,4) for q in original_positions])
    return joints, original_positions


def measure_joint_error(sim, joints, reference):
    return max(abs(float(sim.getJointPosition(j))-float(q)) for j,q in zip(joints,reference))


def get_robot_shape_poses(sim, base):
    """Snap every descendant robot shape w.r.t. model base, to detect falling links."""
    hs=sim.getObjectsInTree(base, sim.object_shape_type, 0)
    if not hs:
        raise RuntimeError('No descendant Franka shapes found; cannot check robot integrity.')
    return {h: [float(x) for x in sim.getObjectPose(h, base)] for h in hs}


def pose_drift(reference, current):
    largest_xyz=0.0; largest_deg=0.0
    if set(reference)!=set(current):
        raise RuntimeError('Robot model shape hierarchy changed during idle test.')
    for h,a in reference.items():
        b=current[h]
        d=math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))
        dot=abs(sum(a[i]*b[i] for i in range(3,7)))
        angle=math.degrees(2.0*math.acos(min(1.0,max(-1.0,dot))))
        largest_xyz=max(largest_xyz,d)
        largest_deg=max(largest_deg,angle)
    return largest_xyz, largest_deg


def preflight_idle_hold(sim, joints, reference, steps=8, tolerance=0.004):
    if sim.getSimulationState()!=sim.simulation_stopped:
        raise RuntimeError('STOP simulation before idle test.')
    # In contrast to V3, verifying only joint angles is NOT enough: links can
    # separate even if the recorded joint angles remain constant.
    base=sim.getObjectParent(joints[0])
    before_shapes=get_robot_shape_poses(sim,base)
    max_q=0.0; max_xyz=0.0; max_angle=0.0
    sim.setStepping(True)
    sim.startSimulation()
    try:
        for _ in range(steps):
            sim.step()
            max_q=max(max_q,measure_joint_error(sim,joints,reference))
            drift, rot=pose_drift(before_shapes,get_robot_shape_poses(sim,base))
            max_xyz=max(max_xyz,drift)
            max_angle=max(max_angle,rot)
    finally:
        sim.stopSimulation()
        for _ in range(100):
            if sim.getSimulationState()==sim.simulation_stopped: break
            time.sleep(0.05)
        else:
            raise RuntimeError('CoppeliaSim did not stop after idle test.')
    print(f'IDLE CHECK: joint drift {max_q:.6f} rad; '
          f'link translation drift {max_xyz*1000:.3f} mm; '
          f'link rotation drift {max_angle:.4f} deg')
    if max_q>tolerance or max_xyz>0.003 or max_angle>0.3:
        raise RuntimeError('Robot changed shape/pose during idle check. '
                           'Do NOT execute. Reload a pristine Franka model and inspect the hierarchy.')
    print('IDLE CHECK PASSED: robot remained assembled for the tested simulation steps.')
    return max_q


def main():
    import argparse
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-base',default='/Franka')
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    sim=RemoteAPIClient().require('sim')
    base=sim.getObject(args.robot_base)
    joints,q=hold_arm_kinematically(sim,base)
    if args.check:
        preflight_idle_hold(sim,joints,q)
    print('Kinematic-only demo configuration completed.')
if __name__=='__main__': main()
