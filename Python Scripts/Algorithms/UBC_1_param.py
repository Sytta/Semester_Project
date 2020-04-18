import numpy as np
import random
import os
import json
from datetime import date

m_max_er = "max_er"
m_min_er = "min_er"
m_er = "er"
m_num_states = "num_states"
m_discount_factor = "discount_factor"
m_q_table = "q_table"
m_max_lr = "max_lr"
m_min_lr = "min_lr"
m_lr = "lr"
m_action_times = "action_times"
m_c = "c"
m_max_times: str = "max_times"
m_times = "times"
m_i = "i"
m_top_max = "top_max"
m_same_action = "same_action"
m_subjectId = "subjectId"



class UCB:
    def __init__(self, num_states, max_er, min_er, discount_factor, max_lr, min_lr, distortion, MAX_TIMES, subjectId, q_table=None, c=2, filename=None):
        self.distortion = distortion # cannot be saved
        if filename is None:
            self.max_exploration_rate = max_er
            self.min_exploration_rate = min_er
            self.exploration_rate = max_er
            self.num_states = num_states
            self.discount_factor = discount_factor
            self.subjectId = subjectId

            if q_table is None:
                self.q_table = np.zeros(num_states)
            else:
                self.q_table = q_table

            self.max_lr = max_lr
            self.min_lr = min_lr
            self.lr = max_lr
            # UBC
            self.action_times = np.zeros(num_states)
            self.c = c
            self.MAX_TIMES = MAX_TIMES
            # Optimize params
            self.times = 0
            self.i = 0
            self.top_max = -1
            self.same_action = 0
        else:
            self.load_model(filename)

    def load_model(self, filename):
        print("Reading UCB model : {}".format(filename))
        with open(filename, 'r') as infile:
            data = json.load(infile)
            self.max_exploration_rate = data[m_max_er]
            self.min_exploration_rate = data[m_min_er]
            self.exploration_rate = data[m_max_er]
            self.num_states = data[m_num_states]
            self.discount_factor = data[m_discount_factor]

            self.q_table = np.asarray(data[m_q_table])

            self.max_lr = data[m_max_lr]
            self.min_lr = data[m_min_lr]
            self.lr = data[m_max_lr]
            # UBC
            self.action_times = np.asarray(data[m_action_times])
            self.c = data[m_c]
            self.MAX_TIMES = data[m_max_times]
            # Optimize params
            self.times = data[m_times]
            self.i = data[m_i]
            self.top_max = data[m_top_max]
            self.same_action = data[m_same_action]
            self.subjectId = data[m_subjectId]

    def select_action(self, t):
        """Select action"""
        if np.random.rand(1) < self.exploration_rate or t == 0:
            # choose randomly
            action = random.randint(0, self.num_states - 1)
        else:
            # UBC
            confidence_bound = self.q_table + self.c * np.sqrt(np.log(t) / (self.action_times + 0.1))  # add 0.1 to avoid division by zero
            action = np.argmax(confidence_bound)

        return action

    def update_Qtable(self, action, reward):
        self.action_times[action] += 1
        old_Q = self.q_table[action]
        self.q_table[action] += self.lr * (reward - old_Q)

    def decay_learning_rate(self, t):
        """Decay over time"""
        lr = self.max_lr - np.log10((t + 1) / 40)
        self.lr = max(self.min_lr, min(self.max_lr, lr))

    def decay_exploration_rate(self, t):
        "Decay over time"
        e_r = self.max_exploration_rate - np.log10((t + 1) / 20)
        self.exploration_rate = max(self.min_exploration_rate, min(self.max_exploration_rate, e_r))

    def save_model(self):
        data = dict({
            m_max_er : float(self.max_exploration_rate),
            m_min_er : float(self.min_exploration_rate),
            m_er : float(self.exploration_rate),
            m_num_states: int(self.num_states),
            m_discount_factor : float(self.discount_factor),
            m_q_table : self.q_table.tolist(),
            m_max_lr : float(self.max_lr),
            m_min_lr : float(self.min_lr),
            m_lr : float(self.lr),
            m_action_times : self.action_times.tolist(),
            m_c: int(self.c),
            m_max_times: int(self.MAX_TIMES),
            m_times : int(self.times),
            m_i: int(self.i),
            m_top_max: int(self.top_max),
            m_same_action: int(self.same_action),
            m_subjectId: int(self.subjectId)
        })

        filename = "../Models/UCB_Subject_{}_{}_trials".format(self.subjectId, self.times, date.today())
        if self.times > 1:
            prev_file = "../Models/UCB_Subject_{}_{}_trials.json".format(self.subjectId, self.times - 1, date.today())
            try:
                os.remove(prev_file)
            except:
                print("Couldn't remove {}".format(prev_file))

        print("Saving model {}".format(filename))
        with open('{}.json'.format(filename), 'w') as outfile:
            json.dump(data, outfile)

    def optimize_policy(self):
        while self.same_action < 15 and self.times < self.MAX_TIMES:
            # select action
            action = self.select_action(self.times)

            # get new state observation
            reward, terminate = self.distortion.step(action)

            # save reward
            self.times += 1

            # update_Qtable
            self.update_Qtable(action, reward)

            if self.i > 35:
                current_max = np.argmax(self.q_table)
                if current_max == self.top_max:
                    self.same_action += 1
                else:
                    self.same_action = 0
                    self.top_max = current_max

            if terminate:
                self.same_action = 0
                self.top_max = -1
                self.save_model()
                continue

            # update rates
            self.decay_exploration_rate(self.i)
            self.decay_learning_rate(self.i)

            self.i += 1
            self.save_model()

        return self.times, self.same_action >= 15
