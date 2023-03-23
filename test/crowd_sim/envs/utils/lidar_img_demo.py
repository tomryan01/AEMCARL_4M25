import unittest
import configparser
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.lidar import *
from crowd_sim.envs.utils.action import ActionRot
from test.test_configs.get_configs import get_config


class LidarImgDemo:
    def __init__(self):
        self._reset()

    def _reset(self):
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

    def test_construct_img(self):
        self._reset()
        scans = []
        vx = 0.
        w = 0.1
        res = 360
        for i in range(100):
            self.robot.step(ActionRot(vx, w))
            scans.append(scan_lidar(self.robot, self.humans, res))

        return construct_img(scans)


if __name__ == '__main__':
    demo = LidarImgDemo()
    img = demo.test_construct_img()
    plt.imshow(img)
    plt.show()
