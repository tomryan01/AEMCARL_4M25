from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState, JointStateLidar


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        # print("self.policy.last_state: ",self.policy.last_state)
        return action


class RobotLidar(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob, lidar_image):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointStateLidar(self.get_full_state(), lidar_image, ob)
        action = self.policy.predict(state)
        # print("self.policy.last_state: ",self.policy.last_state)
        return action