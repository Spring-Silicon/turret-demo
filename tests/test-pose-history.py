import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spring_turret.pose_history import interpolate_pose

def sample(start):
    return {'sampled_at':start,'read_completed_at':start+.032,'axes':{
        name:{'id':sid,'direction':1,'origin':origin,'observed_at':start+offset,
              'degrees':100*(start+offset-10),'goal_degrees':20}
        for name,sid,origin,offset in [('x',2,1005,.003),('y',1,22,.020)]}}

class PoseHistoryTests(unittest.TestCase):
    def test_individually_timed_axes_interpolate_constant_velocity(self):
        history=[sample(10+i*.04) for i in range(5)]
        result=interpolate_pose(history,10.09,10.195)
        self.assertTrue(result['interpolated'])
        self.assertEqual(result['sampled_at'],10.09)
        for axis in result['axes'].values():
            self.assertAlmostEqual(axis['degrees'],9,places=8)
        self.assertGreater(result['read_completed_at'],result['sampled_at'])
        result['axes']['x']['degrees']=999
        self.assertNotEqual(history[2]['axes']['x']['degrees'],999)

    def test_no_extrapolation_missing_brackets_or_stale_bus(self):
        history=[sample(10+i*.04) for i in range(3)]
        for at,now in [(10,10.12),(10.11,10.12),(10.2,10.12),(10.06,10.3)]:
            self.assertIsNone(interpolate_pose(history,at,now))
        self.assertIsNone(interpolate_pose(history,float('nan'),10.12))
        self.assertIsNone(interpolate_pose([],10,10.1))

    def test_no_interpolation_across_recalibration_or_large_sample_gap(self):
        history=[sample(10),sample(10.04)]
        for key,value in [('origin',1050),('id',8),('direction',-1)]:
            bad=copy.deepcopy(history); bad[1]['axes']['x'][key]=value
            self.assertIsNone(interpolate_pose(bad,10.025,10.075))
        self.assertIsNone(interpolate_pose([sample(10),sample(10.2)],10.1,10.235))

    def test_goal_comes_from_past_sample_not_future_command(self):
        history=[sample(10),sample(10.04)]
        history[1]['axes']['x']['goal_degrees']=30
        result=interpolate_pose(history,10.025,10.075)
        self.assertEqual(result['axes']['x']['goal_degrees'],20)

if __name__=='__main__': unittest.main()
