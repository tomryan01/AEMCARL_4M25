import numpy as np
import torch
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import JointStateLidar
from crowd_sim.envs.utils.lidar import construct_img
from crowd_nav.policy.actenvcarl import ACTENVCARL
from crowd_nav.embedding.lidar_embedding_cnn import run


class LidarPolicy(ACTENVCARL):

    def __init__(self, res):
        super().__init__()
        self.name = "LidarPolicy"
        self.res = res
        self.embedding_length = run(construct_img(torch.zeros(res, 100))).shape[-1]

    def input_dim(self):
        return self.self_state_dim + self.embedding_length

    def predict(self, state):
        """
        An update to the 'predict' method of MultiHumanRL that takes a set of LiDAR readings and makes
        predictions based on these, rather than human states

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if not isinstance(state, JointStateLidar):
            raise TypeError('State must be JointSTateLidar to use a LidarPolicy')

        if state.lidar_state.res != self.res:
            raise ValueError('JointStateLidar.lidar_state.res must equal LidarPolicy.res')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            need_build_map = True
            single_step_cnt = 0
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)

                if self.query_env:
                    next_human_states, reward, done, info = self.env.onestep_lookahead(action, need_build_map)
                    need_build_map = False
                else:
                    next_human_states = [
                        self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                        for human_state in state.human_states
                    ]
                    self.map.build_map_cpu(next_human_states)
                    collision_probability = self.map.compute_occupied_probability(next_self_state)
                    reward = self.compute_reward(next_self_state, next_human_states, collision_probability)

                embedding = run(construct_img(torch.tensor(state.lidar_state.readings)))

                batch_states = torch.cat([
                    torch.Tensor(list(next_self_state)).to(self.device),
                    embedding.squeeze(0).to(self.device)
                ],
                    dim=0).unsqueeze(0).unsqueeze(0)

                # VALUE UPDATE

                value, score = self.model(batch_states)
                next_state_value = value.data.item()

                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value

                self.action_values.append(value)
                # ionly save the step cnt of used action
                if value > max_value:
                    max_value = value
                    max_action = action
                    if self.use_rate_statistic:
                        single_step_cnt = self.model.get_step_cnt()

            if max_action is None:
                raise ValueError('Value network is not well trained. ')
            if self.use_rate_statistic:
                if single_step_cnt < len(self.statistic_info):
                    self.statistic_info[single_step_cnt] += 1
                else:
                    print("step count too large!!")
                pass

        if self.phase == 'train':
            self.last_state = self.transform(state)
        return max_action
