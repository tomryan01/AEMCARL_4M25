import logging
import sys
import numpy as np
# configure logging
log_file = "output_test_lidar_10_humans.log"
mode = "w"
file_handler = logging.FileHandler(log_file, mode=mode)
stdout_handler = logging.StreamHandler(sys.stdout)
level = logging.INFO
logging.basicConfig(
    level=level,
    handlers=[stdout_handler, file_handler],
    format="%(asctime)s, %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
import copy
import torch
from crowd_sim.envs.utils.info import *
import time as t



class ExplorerTest(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    # evaluate across k test cases
    def run_k_episodes(self,
                       k,
                       phase,
                       update_memory=False,
                       imitation_learning=False,
                       episode=None,
                       print_failure=False
                       ):
        # logging("run episodes: %d"%(k))
        
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close_ratios = []
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        time_begin = 0
        all_path_len = []
        
        total_test_cases = 10000
        step_size = total_test_cases // k
        for i in range(0, total_test_cases, step_size): # i selects test case
            time_begin = t.time()
            ob = self.env.reset(phase, i)
            last_pos = np.array([self.robot.px, self.robot.py])
            done = False
            states = []
            actions = []
            rewards = []
            path_len = 0.
            stepCounter = 0
            while not done:
                stepCounter += 1
                action = self.robot.act(ob)
                ob, reward, done, info = self.env.step(action)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

                path_len += np.linalg.norm(np.array([self.robot.px, self.robot.py]) - last_pos)
                last_pos = self.robot.get_position()

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            # updatememory中传入的相当于是轨迹，在updatememory中计算state的value，然后存入 memory buffer 中
            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(
                sum([
                    pow(self.gamma, t * self.robot.time_step * self.robot.v_pref) * reward
                    for t, reward in enumerate(rewards)
                ]))
            
            all_path_len.append(path_len)
            too_close_ratios.append(too_close/stepCounter)

            print("i/k: %d/%d, robot pos: %s, %s, time consuming is:%s secs. %s " %
                  (i, k, self.robot.px, self.robot.py, t.time() - time_begin, info.__str__()))

            

        if hasattr(self.robot.policy, "use_rate_statistic"):
            if self.robot.policy.use_rate_statistic:
                statistic_info = self.robot.policy.get_act_statistic_info()
                print("ACT Usage: ", statistic_info)


        # print(self.robot.policy.model.actf_enncoder.act_fn.act_cnt)
        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit
        

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        # print("all_path_len",all_path_len)
        logging.info(
            '{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}, path length: {:.2f}, intrusion time ratio (ITR): {:.2f}, and social distance during intrusions (SD): {:.2f}'.format(
                phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time, average(cumulative_rewards), average(all_path_len), average(too_close_ratios), average(min_dist)))
        # logging.info(
        #     '{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}, path length: {:.2f}, average intrusion ratio: {:.2f}%, '
        # 'average minimal distance during intrusions: {:.2f}'.format(
        #         phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time, average(cumulative_rewards), np.mean(all_path_len),
        #            np.mean(too_close_ratios), np.mean(min_dist)))

        # logging.info('Path length: %.2f Intrusion time ratio (ITR): %.2f and average min separate distance in danger / social distance during intrusions (SD): %.2f',
        #                 average(all_path_len), average(too_close_ratios), average(min_dist))
        # if phase in ['val', 'test']:
        #     total_time = sum(success_times + collision_times + timeout_times) * self.robot.time_step
        #     logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
        #                  too_close / total_time, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)  #CADRL.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([
                    pow(self.gamma,
                        max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward * (1 if t >= i else 0)
                    for t, reward in enumerate(rewards)
                ])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    action_value, _ = self.target_model(next_state.unsqueeze(0))
                    value = reward + gamma_bar * action_value.data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            self.memory.push((state, value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
