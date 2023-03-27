import unittest
import torch
import configparser
from crowd_nav.policy.lidar_policy import LidarPolicy
from crowd_nav.embedding.lidar_embedding_cnn import run
from crowd_sim.envs.utils.lidar import construct_img, scan_lidar
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.state import JointStateLidar, LidarState, FullState
from test.test_configs.get_configs import get_config


def configure(policy):
    config = configparser.RawConfigParser()
    config.read(get_config('policy.config'))
    config.set("actenvcarl", "multi_process", "average")
    config.set("actenvcarl", "act_fixed", False)
    config.set("actenvcarl", "act_steps", 1)
    policy.configure(config)


class LidarPolicyTest(unittest.TestCase):

    def test_construct(self):
        res = 360
        policy = LidarPolicy(res)
        self.assertIsInstance(policy, LidarPolicy)

        configure(policy)

        embedding_output_length = policy.embedding_length
        self.assertEqual(run(construct_img(torch.zeros(res, 100))).shape[-1], embedding_output_length)

        self.assertEqual(policy.model.input_dim, policy.self_state_dim + embedding_output_length)

    def test_predict(self):
        res = 360
        dt = 0.1
        env_config = configparser.RawConfigParser()
        env_config.read(get_config('env.config'))
        robot = Robot(env_config, 'robot')
        robot.time_step = dt
        robot.set(0., 0., 0., 0., 0., 0., 1., 0.)
        env = CrowdSim()
        env.configure(env_config)
        env.set_robot(robot)
        env.generate_random_human_position(2, 'square_crossing')
        policy = LidarPolicy(res)
        configure(policy)
        policy.set_phase('train')
        policy.set_device(torch.device('cpu'))
        policy.set_epsilon(0.01)
        policy.set_env(env)
        policy.time_step = dt
        robot.set_policy(policy)
        env.reset('train')
        scans = []
        for i in range(100):
            scans.append(scan_lidar(robot, [], res))
        state = JointStateLidar(
            robot.get_full_state(),
            LidarState(res, scans)
        )
        policy.predict(state)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
