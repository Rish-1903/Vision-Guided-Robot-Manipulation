#!/usr/bin/env python3
"""Build a new self-contained CoppeliaSim manipulation scene from an empty scene.

Loads an installed *Franka* .ttm model; adds calibrated-camera geometry,
textured checkerboard, 7-joint-chain TCP dummy; later Tasks 4/5 add colored
blocks and a table. Does NOT use ground-truth poses to estimate camera K or
poses. The robot's initial WORLD placement is merely known workcell geometry.

A Franka model asset MUST exist on the local machine; this script does not
silently replace it with a dummy or a non-Franka robot.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
S2 = HERE/'stages'/'task2'
sys.path.insert(0, str(S2))
from make_checkerboard import generate
from _corner_utils import detect_corners  # importing assures Task 2 is present
from importlib import import_module
capture = import_module('01_capture_coppeliasim')

# WORLD coordinates chosen to reproduce the camera-relative robot arrangement
# where the user successfully calibrated 3D points and preflighted six IK poses.
# This is setup geometry, NOT camera ground truth used to estimate calibration.
ROBOT_WORLD_XYZ = (0.6595, 0.3495, 0.0708)
ROBOT_WORLD_YAW_DEG = -140.0


def maybe_get(sim, name):
    try:
        return sim.getObject(name)
    except Exception:
        return None


def candidate_model_dirs(sim):
    roots = []
    for key in ('COPPELIASIM_ROOT', 'COPPELIASIM_PATH', 'COPPELIASIM_MODEL_DIR'):
        if os.environ.get(key): roots.append(Path(os.environ[key]).expanduser())
    for key in ('stringparam_modeldefaultdir', 'stringparam_application_path',
                'stringparam_app_path'):
        try:
            roots.append(Path(sim.getStringParam(getattr(sim,key))).expanduser())
        except Exception:
            continue
    roots.extend([
        Path('/Applications/coppeliaSim.app/Contents/Resources'),
        Path('/Applications/CoppeliaSim.app/Contents/Resources'),
        Path.home()/'Applications/coppeliaSim.app/Contents/Resources',
        Path.home()/'Applications/CoppeliaSim.app/Contents/Resources',
        Path('/opt/CoppeliaSim'),
    ])
    dirs = []
    for root in roots:
        for d in (root, root/'models', root/'Contents/Resources',
                  root/'Contents/Resources/models', root.parent/'models',
                  root/'robots/non-mobile', root/'models/robots/non-mobile'):
            if d.is_dir() and d not in dirs:
                dirs.append(d)
    return dirs


def locate_franka_model(sim, requested):
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'--franka-model not found: {path}')
        if path.suffix.lower() != '.ttm' and not path.name.endswith('.simmodel.xml'):
            raise ValueError('Expected a CoppeliaSim Franka .ttm or .simmodel.xml model')
        return path
    found = []
    for root in candidate_model_dirs(sim):
        # Prefer conventional robot asset paths, then case-insensitive glob.
        for sub in ('robots/non-mobile/Franka.ttm', 'Franka.ttm',
                    'models/robots/non-mobile/Franka.ttm'):
            file = root/sub
            if file.is_file() and file not in found:
                found.append(file)
        if 'models' in root.parts or 'non-mobile' in root.parts:
            for glob_pattern in ('*ranka*.ttm','*ranka*.simmodel.xml',
                                 'robots/non-mobile/*ranka*.ttm',
                                 'robots/non-mobile/*ranka*.simmodel.xml'):
                for file in root.glob(glob_pattern):
                    if file.is_file() and file not in found:
                        found.append(file)
    if not found:
        checked = '\n'.join('  '+str(d) for d in candidate_model_dirs(sim))
        raise FileNotFoundError(
            'Could not automatically locate the installed Franka .ttm model.\n'
            'On your Mac, try:\n  find /Applications "$HOME/Applications" '
            '-iname "*franka*.ttm" 2>/dev/null\n'
            'Then supply --franka-model /full/path/to/Franka.ttm\n'
            f'Searched model directories:\n{checked}')
    return found[0]


def joint_ancestor_count(sim,h,root):
    depth = 0
    for _ in range(100):
        if h == root: return depth
        if h == -1: return None
        if sim.getObjectType(h) == sim.object_joint_type:
            depth += 1
        h = sim.getObjectParent(h)
    return None


def choose_seventh_joint(sim, root):
    descendants = sim.getObjectsInTree(root, sim.handle_all, 0)
    possibilities = [(joint_ancestor_count(sim,h,root), h) for h in descendants
                     if sim.getObjectType(h) == sim.object_joint_type]
    by_depth = [(d,h) for d,h in possibilities if d == 7]
    if len(by_depth) != 1:
        raise RuntimeError(
            f'The loaded Franka model does not provide exactly one seventh '
            f'serial-chain joint (joint depths: {[d for d,h in possibilities]}). '
            'Use the 7-joint Franka model that worked in Task 6, '
            'or explicitly supply its .ttm using --franka-model.')
    return by_depth[0][1], possibilities


def scene_is_blank(sim):
    for name in ('/Franka','/Task2Camera','/Task2Checkerboard','/T4_Red',
                 '/T4_Green','/T4_Blue'):
        if maybe_get(sim,name) is not None: return False
    # Avoid accidental overwriting of unrelated user work.
    allowed = {'DefaultCamera','XYZCameraProxy','XViewCamera','YViewCamera',
               'ZViewCamera','NXViewCamera','NYViewCamera','NZViewCamera',
               'DefaultLights','Floor'}
    try:
        objects = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
        unexpected = []
        for h in objects:
            full = str(sim.getObjectAlias(h, 2)).lstrip('/')
            top_level = full.split('/')[0]
            if top_level not in allowed and str(sim.getObjectAlias(h)) not in allowed:
                unexpected.append(full)
        if unexpected:
            raise RuntimeError('Scene is not blank. Unexpected objects include: '+
                               ', '.join(unexpected[:12])+
                               '. Open File > New scene or use --reset-scene.')
    except RuntimeError:
        raise
    except Exception:
        # Some installations expose extra background/light objects; only
        # refuse if the main workcell objects already exist.
        pass


def build_new_scene(*, model_path='', reset_scene=False,
                    save_path=None, tcp_offset=(0,0,0.10)):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    client = RemoteAPIClient()
    sim = client.require('sim')
    if sim.getSimulationState()!=sim.simulation_stopped:
        raise RuntimeError('Stop simulation before creating the new scene.')
    model = locate_franka_model(sim, model_path)  # detect *before* resetting
    if reset_scene:
        print('Resetting the scene to the CoppeliaSim default (unsaved changes will be lost).')
        sim.loadScene('')
    else:
        scene_is_blank(sim)

    base = sim.loadModel(str(model))
    sim.setObjectAlias(base, 'Franka')
    quat = Rotation.from_euler('z', ROBOT_WORLD_YAW_DEG, degrees=True).as_quat().tolist()
    sim.setObjectPose(base, [*ROBOT_WORLD_XYZ, *quat], -1)
    joint7, joint_depths = choose_seventh_joint(sim, base)
    # The initial model in user's prior scene had 7 joints. Extra finger
    # joints can be okay if they are not ancestors of our TCP dummy.
    print('Loaded robot:', model, '\nRoot alias:',sim.getObjectAlias(base,2),
          '\nFound seventh arm joint:', sim.getObjectAlias(joint7,2))
    print('Joint depths:', [d for d,h in joint_depths])

    tip = sim.createDummy(.03)
    sim.setObjectAlias(tip,'T6_TCP')
    sim.setObjectParent(tip,joint7,False)
    sim.setObjectPose(tip, [*tcp_offset,0.,0.,0.,1.],joint7)
    assert joint_ancestor_count(sim,tip,base)==7
    print('Created provisional TCP:',sim.getObjectAlias(tip,2),
          '| world position:', sim.getObjectPosition(tip,-1))

    from robot_stability import hold_arm_kinematically, preflight_idle_hold
    joints, hold_q = hold_arm_kinematically(sim,base)
    preflight_idle_hold(sim,joints,hold_q)

    texture=S2/'checkerboard.png'
    generate(texture)
    camera, board=capture.create_workcell(sim,texture)
    sim.setObjectPose(camera,[0.,0.,1.,1.,0.,0.,0.],-1)
    # Disable explicitly handled/synthetic depth disabling if supported.
    try:
        sim.setObjectInt32Param(camera,sim.visionintparam_depthignored,0)
    except Exception:
        pass
    print('Created camera:',sim.getObjectAlias(camera,2),
          '| perspective 960x720 / nominal 60 degrees')
    print('Created checkerboard:',sim.getObjectAlias(board,2))
    if save_path is not None:
        save_path=Path(save_path).resolve()
        save_path.parent.mkdir(parents=True,exist_ok=True)
        sim.saveScene(str(save_path))
        print('Saved new workcell scene:',save_path)
    return {'model_file':str(model),
            'root_alias':'/Franka','camera_alias':'/Task2Camera',
            'tcp_alias':sim.getObjectAlias(tip,2),
            'robot_world_position_m':list(ROBOT_WORLD_XYZ),
            'robot_world_yaw_deg':ROBOT_WORLD_YAW_DEG,
            'camera_world_position_m':[0.,0.,1.],
            'tcp_offset_from_joint7_m':list(tcp_offset),
            'tcp_verified':False,'saved_scene':str(save_path) if save_path else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--franka-model',default='',help='Optional absolute path to Franka .ttm')
    p.add_argument('--reset-scene',action='store_true',help='Discard existing scene, build a new one')
    p.add_argument('--output-scene',type=Path,default=HERE/'fresh_workcell.ttt')
    p.add_argument('--tcp-offset',nargs=3,type=float,default=[0.,0.,.10],metavar=('X','Y','Z'))
    args=p.parse_args()
    result=build_new_scene(model_path=args.franka_model,reset_scene=args.reset_scene,
                           save_path=args.output_scene,tcp_offset=args.tcp_offset)
    (args.output_scene.parent/'fresh_scene_setup.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
