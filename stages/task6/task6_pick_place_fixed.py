#!/usr/bin/env python3
"""Task 6 — open-loop VISION GUIDED Panda motion with honest kinematic attachment.

Perception: LIVE RGB and depth -> Task4 HSV masks -> Task5 RGB-D cuboid pose.
Actions: CoppeliaSim simIK config planning; joint TARGET interpolation; optional
kinematic attach after robot reaches an estimated grasp pose; placement in known
work-cell coordinates. Block poses are NEVER read until outcome evaluation.

This is a kinematic pick-and-place BASELINE, not contact/force-based gripping,
collision-free trajectory planning or closed-loop visual servoing.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from task4_common import (BLOCK_SIZE, BLOCK_Z, COLORS, load_calibration,
                           connect, read_bgr, detect_from_rgb, obj_pose_matrix)
from task5_common import read_depth_m, estimate_pose_from_mask, draw_axes
from importlib import import_module

HERE=Path(__file__).resolve().parent


def T_to_pose(T):
    return np.r_[T[:3,3], Rotation.from_matrix(T[:3,:3]).as_quat()].tolist()


def W_from_base(sim, base):
    return obj_pose_matrix(sim.getObjectPose(base,-1))


def known_place_B(sim, base, xyz_world):
    """Target is a user-designated place in world, NOT an object's true pose."""
    p=np.asarray([*xyz_world,1.0],dtype=float)
    return (np.linalg.inv(W_from_base(sim,base))@p)[:3]


def top_down_R(yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,s,0],[s,-c,0],[0,0,-1]],dtype=float)


def make_tcp_goal(center_B, yaw, tcp_above_center):
    """Desired TCP is above the object's center, with downward tool-Z."""
    T=np.eye(4)
    T[:3,:3]=top_down_R(yaw)
    T[:3,3]=np.asarray(center_B)+np.array([0,0,tcp_above_center])
    return T


def way_above(T,meters):
    out=T.copy(); out[2,3]+=meters; return out


def chain_joints(sim, base, tip):
    """Joint list from base to tip; refuse disconnected paths."""
    found=[]; h=tip; n=0
    while h!=-1 and h!=base and n<100:
        if sim.getObjectType(h)==sim.object_joint_type:
            found.append(h)
        h=sim.getObjectParent(h); n+=1
    if h!=base:
        raise RuntimeError('TCP dummy is not a descendant of /Franka; fix its parent in the scene.')
    found.reverse()
    if len(found)<7:
        raise RuntimeError(f'TCP has only {len(found)} ancestor joints (need >=7). Select a dummy beneath the end effector, not the base.')
    if len(found)>8:
        raise RuntimeError(f'TCP has {len(found)} joints in chain; likely selected a finger-tip with gripper joints. Place a TCP dummy on the tool/palm before finger joints.')
    return found


def find_tip(sim,base,explicit=None):
    if explicit:
        return sim.getObject(explicit)
    objs=sim.getObjectsInTree(base,sim.handle_all,0)
    candidates=[]
    for h in objs:
        if sim.getObjectType(h)!=sim.object_dummy_type:
            continue
        alias=str(sim.getObjectAlias(h)).lower()
        if not any(x in alias for x in ('tcp','iktip','grasp_tip','tool_tip','flange_tip')):
            continue
        try:
            path=chain_joints(sim,base,h)
            candidates.append((h,str(sim.getObjectAlias(h)),len(path)))
        except RuntimeError:
            pass
    if len(candidates)!=1:
        print('Possible tip candidates:',candidates)
        raise RuntimeError('Specify --tip /FULL/PATH to a TCP dummy under the last arm joint; no unique correct tip can be inferred.')
    print('Selected TCP dummy:', candidates[0])
    return candidates[0][0]


def index_map(mapping, sim_handle):
    """ZeroMQ maps may contain numeric or stringified object-handle keys."""
    if sim_handle in mapping: return mapping[sim_handle]
    if str(sim_handle) in mapping: return mapping[str(sim_handle)]
    raise KeyError(f'Could not map scene handle {sim_handle} to IK-world handle')


class IKMotion:
    def __init__(self,client,sim,base,tip, joints):
        self.sim=sim; self.base=base; self.tip=tip; self.joints=joints
        self.ik=client.require('simIK')
        self.target=sim.createDummy(0.012)
        sim.setObjectAlias(self.target,'Task6IKTarget')
        sim.setObjectPose(self.target,sim.getObjectPose(tip,base),base)
        self.env=self.ik.createEnvironment()
        self.group=self.ik.createGroup(self.env)
        el, m, _ = self.ik.addElementFromScene(
            self.env,self.group,base,tip,self.target,self.ik.constraint_pose)
        self.ik.setElementPrecision(self.env,self.group,el,[0.006,math.radians(6)])
        self.ik.setGroupCalculation(self.env,self.group,
                                    self.ik.method_damped_least_squares,0.06,80)
        self.ik_joints=[index_map(m,j) for j in joints]
        self.ik_target=index_map(m,self.target)
        self.ik_base=index_map(m,self.base)

    def plan(self, goal):
        # Synchronize robot state without using any simulator object ground truth.
        self.ik.syncFromSim(self.env,[self.group])
        self.ik.setObjectPose(self.env,self.ik_target,T_to_pose(goal),self.ik_base)
        configs=self.ik.findConfigs(self.env,self.group,self.ik_joints,
            {'maxDist':2.0,'maxTime':2.0,'findMultiple':False})
        if not configs:
            raise RuntimeError('No IK solution. The waypoint may be outside the arm workspace, or TCP axes/tool offset may be wrong.')
        return np.asarray(configs[0],float)

    def move_joints(self, q_goal, label='joint target', resolution=.012):
        """Deterministic kinematic interpolation for the 7 arm joints.

        Does not check collisions; the workcell uses a kinematic grasp proxy.
        This avoids gravity-induced drift and conflicts with dynamic motors.
        """
        q0=np.asarray([self.sim.getJointPosition(j) for j in self.joints],float)
        q1=np.asarray(q_goal,float)
        if q1.shape != q0.shape: raise ValueError('Expected one target per arm joint')
        n=max(30,int(np.max(np.abs(q1-q0))/resolution)+1)
        for i in range(1,n+1):
            alpha=i/n
            alpha=alpha*alpha*(3-2*alpha)
            q=(1-alpha)*q0+alpha*q1
            for j,angle in zip(self.joints,q):
                self.sim.setJointPosition(j,float(angle))
                self.sim.setJointTargetPosition(j,float(angle))
            self.sim.step()
        final=np.asarray([self.sim.getJointPosition(j) for j in self.joints])
        if float(np.max(np.abs(final-q1)))>.03:
            raise RuntimeError(f'{label}: joint tracking failed. Robot model scripts may override kinematic joints.')
        print(f'  {label} reached: max joint error {np.max(np.abs(final-q1)):.5f} rad')
        return final

    def move(self,goal,label='waypoint',resolution=0.025,max_steps=600):
        q1=self.plan(goal)
        self.move_joints(q1,label, resolution=min(.012,resolution))
        tpose=obj_pose_matrix(self.sim.getObjectPose(self.tip,self.base))
        err=float(np.linalg.norm(tpose[:3,3]-goal[:3,3]))
        angle=math.degrees(Rotation.from_matrix(tpose[:3,:3].T@goal[:3,:3]).magnitude())
        if err>=0.018 or angle>=12:
            raise RuntimeError(f'{label}: IK pose tolerance not reached: {1000*err:.1f}mm, {angle:.1f}deg')
        print(f'  {label} TCP pose error {1000*err:.1f}mm, {angle:.1f}deg')
        return {'tip_error_mm':1000*err,'tip_angle_deg':angle}

    def cleanup(self):
        try: self.ik.eraseEnvironment(self.env)
        except Exception: pass
        try: self.sim.removeObject(self.target)
        except Exception: pass


def live_poses(sim,camera,K,dist,B_from_C,output):
    """Capture one live RGB-D frame, including transparent failure diagnostics.

    Saved debug masks and depth are observations, never simulator object poses.
    """
    bgr = read_bgr(sim,camera)
    depth = read_depth_m(sim,camera,bgr.shape[:2])
    detections,masks = detect_from_rgb(bgr)
    cv2.imwrite(str(output/'rgb_before_pick.png'), bgr)
    valid_depth = np.isfinite(depth) & (depth > 0.08) & (depth < 5.0)
    print('RGB shape:', bgr.shape,
          '| range:', int(bgr.min()), '..', int(bgr.max()))
    print('Valid depth pixels:', int(valid_depth.sum()), '/', int(depth.size))
    finite_depth = depth[np.isfinite(depth)]
    if finite_depth.size:
        print('Depth range [m]:', float(finite_depth.min()), '..', float(finite_depth.max()))
        lo, hi = np.percentile(finite_depth, [1,99])
        if hi - lo < 1e-6: hi = lo + 1.0
        normalized = np.clip((depth-lo) / (hi-lo), 0., 1.)
        depth_vis = np.nan_to_num(normalized, nan=0.0).astype(np.float32)
        cv2.imwrite(str(output/'depth_before_pick.png'),
                    cv2.applyColorMap((255*depth_vis).astype(np.uint8), cv2.COLORMAP_TURBO))
    poses={}; diag={}
    overlay=bgr.copy()
    for color in COLORS:
        mask=masks[color]
        cv2.imwrite(str(output/f'mask_{color}.png'), mask)
        mask_count=int(np.count_nonzero(mask))
        depth_count=int(np.count_nonzero((mask > 0) & valid_depth))
        print(f'{color}: HSV mask {mask_count} px, mask + valid depth {depth_count} px')
        T,info=estimate_pose_from_mask(depth,mask,K,dist,B_from_C)
        poses[color]=T; diag[color]=info
        if T is not None:
            draw_axes(overlay,T,B_from_C,K,dist)
        else:
            print(f'  {color} pose unavailable:',info.get('reason','unknown'))
    cv2.imwrite(str(output/'perception_before_pick.png'),overlay)
    return poses,diag


def evaluate_object(sim,obj,base,expected_place_B,estimated_pick_B):
    """ONLY evaluation: object pose is read after manipulation, never in perception/control."""
    gt=obj_pose_matrix(sim.getObjectPose(obj,base))
    pos=gt[:3,3]
    err=float(np.linalg.norm(pos-expected_place_B))*1000
    horizontal=float(np.linalg.norm(pos[:2]-expected_place_B[:2]))*1000
    vertical=1000*float(abs(pos[2]-expected_place_B[2]))
    success=(horizontal<30.0 and vertical<20.0)
    return {'final_position_B_m':pos.tolist(),'placement_error_mm':err,
            'placement_xy_error_mm':horizontal,'placement_z_error_mm':vertical,
            'placement_success_30mm_xy_20mm_z':bool(success),
            'note':'Outcome evaluation uses CoppeliaSim ground truth, not to control motion.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera',default='/Task2Camera')
    p.add_argument('--robot-base',default='/Franka')
    p.add_argument('--tip',default='',help='Full path of TCP dummy below last arm joint. Required unless auto-discovery finds exactly one.')
    p.add_argument('--color',choices=list(COLORS),default='red')
    p.add_argument('--calibration',type=Path,default=HERE.parent/'vision_calibration_task3'/'task3_results'/'recovered'/'camera_to_robot_transform_recovered.npz')
    p.add_argument('--place-world',type=float,nargs=3,metavar=('X','Y','Z'),default=[0.0,-0.20,float(BLOCK_Z)],help='Desired object center in WORLD meters, NOT image/ground-truth pose')
    p.add_argument('--tcp-height',type=float,default=.09,help='TCP offset above cube center at grasp, meters; must be adapted to gripper geometry')
    p.add_argument('--lift-height',type=float,default=.18)
    p.add_argument('--output',type=Path,default=HERE/'task6_results')
    p.add_argument('--setup-demo-layout',action='store_true',
                   help='BEFORE simulation starts: place the three Task4 blocks at known, non-overlapping test positions. These scene-setup positions are NOT used for perception or planning.')
    p.add_argument('--dry-run',action='store_true',help='Only visualize live vision poses and planned grasp. No robot motion.')
    p.add_argument('--execute',action='store_true',help='Run experimental kin-attached pick and place. Requires explicit opt-in.')
    args=p.parse_args()
    if args.execute and args.dry_run: p.error('Choose only one of --dry-run or --execute')
    if not args.execute and not args.dry_run: args.dry_run=True
    if args.tcp_height<.025 or args.lift_height<.05: p.error('Specify a plausible positive TCP offset and lift height')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    K,dist,B_from_C=load_calibration(args.calibration)
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    client=RemoteAPIClient(); sim=client.require('sim')
    base=sim.getObject(args.robot_base)
    camera=sim.getObject(args.camera)
    from task4_common import get_maybe
    if args.setup_demo_layout:
        if sim.getSimulationState() != sim.simulation_stopped:
            raise RuntimeError('Stop the simulation before --setup-demo-layout; layout is set in the initial stopped-scene state.')
        from task4_common import prepare_scene, apply_layout
        objects = prepare_scene(sim)
        apply_layout(sim, objects, np.random.default_rng(42), 0)
        print('Initialized a known, non-overlapping Task 4 test layout BEFORE simulation start:')
        print('  red=(-0.19,-0.11), green=(+0.19,-0.10), blue=(0,+0.135), in WORLD metres.')
        print('The layout is ONLY scene preparation; live camera observations are used to plan the grasp.')
    obj=get_maybe(sim,COLORS[args.color]['alias'])
    if obj is None:
        raise SystemExit(f"Missing {COLORS[args.color]['alias']}. Open the saved Task 5 scene; do not randomize it after starting perception.")
    # JOINT PRECHECK: early, before modifying the simulation.
    tip=find_tip(sim,base,args.tip)
    joints=chain_joints(sim,base,tip)
    print('Articulated IK chain:', [str(sim.getObjectAlias(j)) for j in joints])
    print('Grasp mode: kinematic parenting (not a physically simulated finger grasp)')
    sim.setStepping(True)
    sim.startSimulation()
    motion=None
    attached=False
    result={'color':args.color,'calibration':str(args.calibration),
            'grasp_mode':'KINEMATIC_ATTACH_PROXY',
            'uses_live_RGB_D_for_pick':True,'uses_object_true_pose_for_planning':False,
            'collision_free_planning':False,'success':False}
    try:
        for _ in range(3):sim.step()
        poses,diag=live_poses(sim,camera,K,dist,B_from_C,out)
        color=args.color
        est=poses[color]
        if est is None:
            raise RuntimeError(
                f'Live RGB-D pose for {color} unavailable: {diag[color]}. '
                'Inspect task6_results/rgb_before_pick.png, mask_red.png, '
                'depth_before_pick.png. If mask is blank, move the block '
                'into the camera view or use --setup-demo-layout. If mask '
                'is present but depth has no valid pixels, enable depth '
                'and make sure the vision sensor is being handled.')
        yaw=math.atan2(est[1,0],est[0,0])
        pick_center=est[:3,3].copy()
        place_center=known_place_B(sim,base,args.place_world)
        result['estimated_pick_center_B_m']=pick_center.tolist()
        result['goal_object_center_B_m']=place_center.tolist()
        result['estimated_yaw_deg']=math.degrees(yaw)
        result['perception_diagnostics']=diag[color]
        grasp=make_tcp_goal(pick_center,yaw,args.tcp_height)
        above_pick=way_above(grasp,args.lift_height)
        place=make_tcp_goal(place_center,yaw,args.tcp_height)
        above_place=way_above(place,args.lift_height)
        print(f'Vision-estimated {color} center [base m]:',np.round(pick_center,4))
        print('Target place center [base m]:',np.round(place_center,4))
        if args.dry_run:
            print('DRY-RUN: Did not move joints or attach block. Review estimated center, TCP offset, orientation, and destination.')
            result['status']='dry_run';return
        motion=IKMotion(client,sim,base,tip,joints)
        motion.move(above_pick,'approach above pick')
        motion.move(grasp,'descend to pick')
        # Kinematic attachment is a surrogate grip, never physical force closure.
        # Use object HANDLE for actuation, no object GT pose is queried or used.
        actual_tip=obj_pose_matrix(sim.getObjectPose(tip,base))
        if np.linalg.norm(actual_tip[:3,3]-grasp[:3,3])>.02:
            raise RuntimeError('TCP not at predicted grasp target; refusing to attach block')
        sim.setObjectParent(obj,tip,True)
        attached=True
        for _ in range(4):sim.step()
        motion.move(above_pick,'lift')
        motion.move(above_place,'transfer')
        motion.move(place,'lower to place')
        sim.setObjectParent(obj,-1,True)
        attached=False
        for _ in range(4):sim.step()
        result.update(evaluate_object(sim,obj,base,place_center,est))
        result['status']='executed'
        result['success']=bool(result['placement_success_30mm_xy_20mm_z'])
        motion.move(above_place,'retreat')
        print('PLACEMENT ERROR XY:',round(result['placement_xy_error_mm'],2),'mm')
        print('SUCCESS (XY < 30 mm; Z < 20 mm):',result['success'])
    except Exception as exc:
        result['status']='failed'
        result['error']=str(exc)
        print('ERROR:',exc,file=sys.stderr)
        raise
    finally:
        # Cleanup on all exits. Keep attachment if an execution error occurs
        # (so the object is not incorrectly counted as released).
        if motion is not None: motion.cleanup()
        sim.stopSimulation()
        (out/'trial_result.json').write_text(json.dumps(result,indent=2))
        print('Saved:',out/'trial_result.json')

if __name__=='__main__':main()
