from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        # for o in ob:
        #     print(str(o))
        # ob = []
        state = JointState(self.get_full_state(), ob)
        print(str(state.self_state))
        
        action = self.policy.predict(state)
        # print("self.policy.last_state: ",self.policy.last_state)
        return action
