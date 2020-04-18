import numpy as np
import random
import json
import os
from datetime import date

m_max_er = "max_er"
m_min_er = "min_er"
m_er = "er"
m_num_states_R = "num_states_R"
m_num_states_G = "num_states_G"
m_discount_factor = "discount_factor"
m_q_table_R = "q_table_R"
m_q_table_G = "q_table_G"
m_max_lr = "max_lr"
m_min_lr = "min_lr"
m_lr = "lr"
m_action_times_R = "action_times_R"
m_action_times_G = "action_times_G"
m_c = "c"
m_times = "times"
m_top_max_G = "top_max_G"
m_top_max_R = "top_max_R"
m_same_action = "same_action"
m_subjectId = "subjectId"
m_countTimes = "countTimes"
m_ertimes = "er_times"
m_lrtimes = "lr_times"


class Distortion:
    def __init__(self, init_val, correct_val, R, Gain):
        self.distortion = init_val
        self.correct_val = correct_val
        self.R_max = R[-1]
        self.G_max = Gain[-1]
        self.R = R
        self.G = Gain

        # self.values = []
        # index = 0
        # self.index_values = dict()
        # for r in self.R:
        #     for g in self.G:
        #         self.values.append((r, g))
        #         self.index_values[(r, g)] = index
        #         index += 1

    def step(self, action):
        # (radius, gain) = self.values[action]

        radius = self.R[action[0]]
        gain = self.G[action[1]]

        correct_radius = float(self.correct_val[0])
        correct_gain = float(self.correct_val[1])

        if radius <= correct_radius and gain <= correct_gain:
            return (1, gain), False
        else:
            return (-1, -gain), False


discount_factor = 0.98
exploration_rate = 1
MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.1
learning_rate = 0.5
MAX_LEARNING_RATE = 0.5
MIN_LEARNING_RATE = 0.1
MAX_TIMES = 850


class UCB2:
    def __init__(self, num_states_R, num_states_G, max_er, min_er, discount_factor, max_lr, min_lr, distortion, subjectId, lr_times=180, er_times=100, count_times=160, MAX_TIMES=200, c=2, filename=None):
        self.distortion = distortion
        self.MAX_TIMES = MAX_TIMES

        if filename is None:
            self.max_exploration_rate = max_er
            self.min_exploration_rate = min_er
            self.exploration_rate = max_er
            self.num_states_R = num_states_R
            self.num_states_G = num_states_G
            self.discount_factor = discount_factor

            self.q_table_R = np.zeros(self.num_states_R)
            self.q_table_G = np.zeros(self.num_states_G)

            self.max_lr = max_lr
            self.min_lr = min_lr
            self.lr = max_lr

            self.action_times_R = np.zeros(self.num_states_R)
            self.action_times_G = np.zeros(self.num_states_G)
            self.c = c

            self.lr_times = lr_times
            self.er_times = er_times
            self.count_times = count_times
            self.times = 0
            self.top_max_G = -1
            self.top_max_R = -1
            self.same_action = 0
            self.subjectId = subjectId
        else:
            self.load_model(filename)

    def load_model(self, filename):
        print("Reading UCB2 model : {}".format(filename))
        with open(filename, 'r') as infile:
            data = json.load(infile)
            self.max_exploration_rate = data[m_max_er]
            self.min_exploration_rate = data[m_min_er]
            self.exploration_rate = data[m_max_er]
            self.num_states_G = data[m_num_states_G]
            self.num_states_R = data[m_num_states_R]
            self.discount_factor = data[m_discount_factor]

            self.q_table_G = np.asarray(data[m_q_table_G])
            self.q_table_R = np.asarray(data[m_q_table_R])

            self.max_lr = data[m_max_lr]
            self.min_lr = data[m_min_lr]
            self.lr = data[m_max_lr]
            # UBC
            self.action_times_R = np.asarray(data[m_action_times_R])
            self.action_times_G = np.asarray(data[m_action_times_G])

            self.c = data[m_c]
            self.count_times = data[m_countTimes]

            # Optim params
            self.times = data[m_times]
            self.top_max_G = data[m_top_max_G]
            self.top_max_R = data[m_top_max_R]
            self.same_action = data[m_same_action]
            self.subjectId = data[m_subjectId]
            self.er_times = data[m_ertimes]
            self.lr_times = data[m_lrtimes]

    def select_action(self, t):
        """Select action"""
        if np.random.rand(1) < self.exploration_rate or t == 0:
            # choose randomly
            R = random.randint(0, self.num_states_R - 1)
            G = random.randint(0, self.num_states_G - 1)
        else:
            # UBC
            confidence_bound_R = self.q_table_R + self.c * np.sqrt(
            np.log(t) / (self.action_times_R + 0.1))  # add 0.1 to avoid division by zero
            R = np.argmax(confidence_bound_R)

            confidence_bound_G = self.q_table_G + self.c * np.sqrt(
            np.log(t) / (self.action_times_G + 0.1))  # add 0.1 to avoid division by zero
            G = np.argmax(confidence_bound_G)

        return (R, G)

    def update_Qtable(self, action, reward):
        self.action_times_R[action[0]] += 1
        self.action_times_G[action[1]] += 1

        old_Q_R = self.q_table_R[action[0]]
        old_Q_G = self.q_table_G[action[1]]

        self.q_table_R[action[0]] += self.lr * (reward[0] - old_Q_R)
        self.q_table_G[action[1]] += self.lr * (reward[1] - old_Q_G)

    def decay_learning_rate(self, t):
        """Decay over time"""
        lr = self.max_lr - np.log10((t + 1) / self.lr_times)
        self.lr = max(self.min_lr, min(self.max_lr, lr))

    def decay_exploration_rate(self, t):
        "Decay over time"
        e_r = self.max_exploration_rate - np.log10((t + 1) / self.er_times)
        self.exploration_rate = max(self.min_exploration_rate, min(self.max_exploration_rate, e_r))

    def save_model(self):
        data = dict({
            m_max_er: float(self.max_exploration_rate),
            m_min_er: float(self.min_exploration_rate),
            m_er: float(self.exploration_rate),
            # m_num_states: int(self.num_states),
            m_num_states_R: int(self.num_states_R),
            m_num_states_G: int(self.num_states_G),
            m_discount_factor: float(self.discount_factor),
            m_q_table_R: self.q_table_R.tolist(),
            m_q_table_G: self.q_table_G.tolist(),
            m_max_lr: float(self.max_lr),
            m_min_lr: float(self.min_lr),
            m_lr: float(self.lr),
            m_action_times_R: self.action_times_R.tolist(),
            m_action_times_G: self.action_times_G.tolist(),
            m_c: int(self.c),
            # m_max_times: int(self.MAX_TIMES),
            m_ertimes : float(self.er_times),
            m_lrtimes : float(self.lr_times),
            m_times: int(self.times),
            m_top_max_R: int(self.top_max_R),
            m_top_max_G: int(self.top_max_G),
            m_same_action: int(self.same_action),
            m_subjectId: int(self.subjectId),
            m_countTimes : int(self.count_times)
        })

        filename = "../Models/UCB2_Subject_{}_{}_trials".format(self.subjectId, self.times, date.today())
        if self.times > 1:
            prev_file = "../Models/UCB2_Subject_{}_{}_trials.json".format(self.subjectId, self.times - 1, date.today())
            try:
                os.remove(prev_file)
            except:
                print("Couldn't remove {}".format(prev_file))

        print("Saving model {}".format(filename))
        with open('{}.json'.format(filename), 'w') as outfile:
            json.dump(data, outfile)

    def optimize_policy(self):

        while self.same_action < 30 and self.times < self.MAX_TIMES:

            # select action
            action = self.select_action(self.times)

            # get new state observation
            reward, terminate = self.distortion.step(action)

            # update_Qtable
            self.update_Qtable(action, reward)

            if self.times > self.count_times:
                current_max_G = np.argmax(self.q_table_G)
                current_max_R = np.argmax(self.q_table_R)

                if current_max_G == self.top_max_G and current_max_R == self.top_max_R:
                    self.same_action += 1
                else:
                    self.same_action = 0
                    self.top_max_G = current_max_G
                    self.top_max_R = current_max_R

            # update rates
            self.decay_exploration_rate(self.times)
            self.decay_learning_rate(self.times)

            self.times += 1
            self.save_model()

        return self.times, self.same_action >= 30