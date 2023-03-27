import numpy as np
import torch
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import JointStateLidar
from crowd_sim.envs.utils.lidar import construct_img
from crowd_nav.policy.actenvcarl import ACTENVCARL
from crowd_nav.embedding.lidar_embedding_cnn import run
from crowd_nav.common.qnetwork_factory import ValueNetworkBase


class LidarPolicy(ACTENVCARL):

    def __init__(self):
        super().__init__()
        self.name = "LidarPolicy"


    def configure(self, config):
        self.set_common_parameters(config)
        lidar_input_dim = [int(x) for x in config.get("lidar", "lidar_input_dim").split(", ")] 
        in_mlp_dims = [int(x) for x in config.get("lidar", "in_mlp_dims").split(", ")]
        # gru_hidden_dim = [int(x) for x in config.get('actenvcarl', 'gru_hidden_dim').split(', ')]
        sort_mlp_dims = [int(x) for x in config.get("lidar", "sort_mlp_dims").split(", ")]
        sort_mlp_attention = [int(x) for x in config.get("lidar", "sort_attention_dims").split(", ")]
        # aggregation_dims = [int(x) for x in config.get('actenvcarl', 'aggregation_dims').split(', ')]
        action_dims = [int(x) for x in config.get("lidar", "action_dims").split(", ")]
        self.with_om = config.getboolean("lidar", "with_om")
        with_dynamic_net = config.getboolean("lidar", "with_dynamic_net")
        with_global_state = config.getboolean("lidar", "with_global_state")
        test_policy_flag = [int(x) for x in config.get("lidar", "test_policy_flag").split(", ")]
        multi_process_type = config.get("lidar", "multi_process")

        act_fixed = config.get("lidar", "act_fixed")
        act_steps = config.get("lidar", "act_steps")

        print("process type ", type(multi_process_type))
        print(type(test_policy_flag[0]))
        print("act steps: ", act_steps," act_fixed: ",act_fixed)
        # def __init__(self, input_dim, self_state_dim, joint_state_dim, in_mlp_dims, sort_mlp_dims, action_dims, with_dynamic_net=True):

        print(
            self.self_state_dim,
            self.joint_state_dim,
            in_mlp_dims,
            sort_mlp_dims,
            sort_mlp_attention,
            action_dims
        )
        self.model = ValueNetworkBase(self.input_dim(),
                                      self.self_state_dim,
                                      self.joint_state_dim,
                                      in_mlp_dims,
                                      sort_mlp_dims,
                                      sort_mlp_attention,
                                      action_dims,
                                      with_dynamic_net,
                                      with_global_state,
                                      test_policy_flag[0],
                                      multi_process_type=multi_process_type,
                                      act_steps=act_steps,
                                      act_fixed=act_fixed,
                                      lidar_input_dim = lidar_input_dim).product()

        self.multiagent_training = config.getboolean("actcarl", "multiagent_training")
        # logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))
        # logging.info('Policy: {} {} interaction state'.format(self.name, 'w/' if with_interaction else 'w/o'))


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

        # if state.lidar_state.res != self.res:
        #     raise ValueError('JointStateLidar.lidar_state.res must equal LidarPolicy.res')

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
                    next_human_states, next_lidar_image, reward, done, info = self.env.onestep_lookahead(action, need_build_map)
                    need_build_map = False
                    print(type(next_lidar_image))
                else:
                    next_human_states = [
                        self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                        for human_state in state.human_states
                    ]
                    self.map.build_map_cpu(next_human_states)
                    collision_probability = self.map.compute_occupied_probability(next_self_state)
                    reward = self.compute_reward(next_self_state, next_human_states, collision_probability)

                next_self_state = torch.Tensor(next_self_state.get_state()).to(self.device)
                next_lidar_image = next_lidar_image.to(self.device)
                rotate_next_self_state = self.rotate(next_self_state.unsqueeze(0))
                print(rotate_next_self_state.size())

                # VALUE UPDATE

                value, score = self.model(rotate_next_self_state, next_lidar_image.unsqueeze(0))
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

        # if self.phase == 'train':
        #     self.last_state = self.transform(state)
        return max_action
    
    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
        #  0     1      2     3      4        5     6      7         8     
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)

        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy], dim=1)
        return new_state
