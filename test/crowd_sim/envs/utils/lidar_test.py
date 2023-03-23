import unittest
import configparser
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.lidar import *
from crowd_sim.envs.utils.action import ActionRot
from test.test_configs.get_configs import get_config
import matplotlib.pyplot as plt


class LidarTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(LidarTest, self).__init__(*args, **kwargs)
        self.res = 4
        env_config = configparser.RawConfigParser()
        env_config.read(get_config('env.config'))
        human_positions = [[1, 0], [2, 0], [-1, 0], [-2, 0]]
        self.humans = []
        self.robot = Robot(env_config, 'robot')
        self.robot.time_step = 0.1
        self.robot.set_position([0, 0])
        self.robot.theta = 0.
        for i in range(4):
            self.humans.append(Human(env_config, 'humans'))
            self.humans[i].set_position(human_positions[i])

    def test_scan_lidar(self):
        scan = scan_lidar(self.robot, self.humans, self.res)
        self.assertListEqual(list(scan), [1., np.inf, 1., np.inf])

    def test_scan_to_points(self):
        scan = scan_lidar(self.robot, self.humans, self.res)
        points = scan_to_points(scan, self.robot, self.res)
        self.assertTrue(np.linalg.norm(np.array(points) - np.array([[1., 0.], [-1., 0.]])) < 0.01)


if __name__ == '__main__':
    unittest.main()
