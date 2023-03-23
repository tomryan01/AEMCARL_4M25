import unittest
import configparser
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.lidar import *
from crowd_sim.envs.utils.action import ActionRot
from test.test_configs.get_configs import get_config
from crowd_nav.embedding.lidar_embedding_cnn import run


class LidarEmbeddingCNNTest(unittest.TestCase):
    """
    There isn't much testing that can be done on the embedding
    output, but this ensures we can do a forward pass of the
    network without errors
    """
    def test_embedding_cnn(self):
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

        scans = []
        vx = 0.
        w = 0.1
        res = 360
        for i in range(100):
            self.robot.step(ActionRot(vx, w))
            scans.append(scan_lidar(self.robot, self.humans, res))

        img = construct_img(scans)

        network_output = run(img)

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
