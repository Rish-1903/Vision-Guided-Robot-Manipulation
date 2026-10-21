import unittest
from robot_stability import hold_arm_kinematically, measure_joint_error, pose_drift

class FakeSim:
    simulation_stopped=0
    object_joint_type=1
    jointmode_kinematic=4
    handle_scene=-999
    def __init__(self):
        self.q={j:0.1*j for j in range(1,8)}
        self.modes={}
        self.props={self.handle_scene:{'dynamicsEnabled': True}}
        self.calls=[]
    def getSimulationState(self): return 0
    def setBoolProperty(self,h,p,v):
        self.calls.append(('prop',h,p,v))
        self.props.setdefault(h,{})[p]=v
    def getBoolProperty(self,h,p):return self.props[h][p]
    def getObjectsInTree(self,base,typ,flags):return list(range(1,8))
    def getObjectType(self,h):return self.object_joint_type
    def getObjectParent(self,h):return h-1
    def getJointPosition(self,j):return self.q[j]
    def setJointMode(self,j,mode):
        assert self.props[self.handle_scene]['dynamicsEnabled'] is False
        self.calls.append(('jointmode',j,mode))
        self.modes[j]=mode
    def setJointPosition(self,j,q):self.q[j]=q
    def setJointTargetPosition(self,j,q):pass

class HoldTests(unittest.TestCase):
    def test_disables_scene_physics_first(self):
        s=FakeSim()
        joints,q=hold_arm_kinematically(s,0,log=False)
        self.assertEqual(len(joints),7)
        self.assertFalse(s.getBoolProperty(s.handle_scene,'dynamicsEnabled'))
        self.assertLess(next(i for i,c in enumerate(s.calls) if c[2:3]==('dynamicsEnabled',)),
                        next(i for i,c in enumerate(s.calls) if c[0]=='jointmode'))
        self.assertEqual(len(s.modes),7)
        self.assertAlmostEqual(measure_joint_error(s,joints,q),0)
    def test_rigid_shape_drift(self):
        ref={2:[0,0,0,0,0,0,1]}
        self.assertAlmostEqual(pose_drift(ref,{2:[0,0,0,0,0,0,1]})[0],0)
        self.assertGreater(pose_drift(ref,{2:[0,0.02,0,0,0,0,1]})[0],0.01)

if __name__=='__main__':unittest.main()
