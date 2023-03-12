from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState, JointStateLidar
import matplotlib.pyplot as plt


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob, lidar_image):
        # print(lidar_image.shape)
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        # for o in ob:
        #     print(str(o))
        # ob = []
        # plt.imshow(lidar_image)
        # plt.show()
        # print(str(state.self_state))
        state = JointState(self.get_full_state(), ob)
        
        action = self.policy.predict(state, lidar_image)
        # print("self.policy.last_state: ",self.policy.last_state)
        return action
