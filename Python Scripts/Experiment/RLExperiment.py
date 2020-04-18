from Communication.Client import TCP
from Algorithms.UBC_1_param import UCB
import numpy as np
from Logger.RLLogger import *


class Distortion:
    def __init__(self, init_val, values, tcp, logger):
        self.init = init_val
        self.distortion = init_val
        self.values = values
        self.tcp = tcp
        self.logger = logger
        self.max = self.values[-1]

        # Construct rewards count
        self.construct_rewards_count()

    def construct_rewards_count(self):
        # Prepare to keep track of the all the rewards received
        neg_rewards = [-val for val in self.values]
        all_rewards = self.values + neg_rewards
        self.rewards_count = dict.fromkeys(all_rewards, 0)

    def correct_reward(self, reward, terminate, detected):
        # Correct reward
        # Terminate is just a parameter for the UCB algorithm, not really important, you can ignore it
        # Eg: for the past, for gain 5, if the subject never noticed, and this time he detected, we modify
        # his reaction to have not detected as well
        self.rewards_count[reward] += 1
        pos_count = self.rewards_count[abs(reward)]
        neg_count = self.rewards_count[-abs(reward)]

        detected_corrected = detected

        # If he has detected the gain most of the time
        if pos_count > neg_count:
            r = abs(reward)
            detected_corrected = False
        # If he has not detected the gain most of the time
        elif neg_count > pos_count:
            r = -abs(reward)
            detected_corrected = True
        else:
            # If the number of detection = number of no detection, do not modify the reward
            if detected:
                r = -abs(reward)
            else:
                r = abs(reward)

        self.logger.logUCB(abs(reward), detected, detected_corrected)

        print("Detected {} - Original Reward: {}, Corrected Reward: {}, Terminate: {}".format(detected, reward, r, terminate))
        return r, terminate

    def step(self, action):
        # UCB calls this function
        # Get the distortion from the subject, correct the reward, and send it to the UCB algorithm

        # Choose the distortion depending on the action UCB picked
        self.distortion = self.values[action]
        gain = self.distortion
        if gain > self.max: # verify that it didn't exceed the max distortion
            gain = self.max

        # Send the gain to the subject and get his response
        self.tcp.send(gain)
        answers = self.tcp.receive()
        distortion = answers['distortion']
        detected = answers['detected']

        assert(gain == distortion) # "Gain {} must be equal to distortion {}!".format(gain, distortion))

        terminate = (self.distortion > self.max or self.distortion < self.values[1]) #0.25

        # Send corrected reward to UCB
        if detected:
            return self.correct_reward(-self.distortion, terminate, detected)
        else:
            return self.correct_reward(self.distortion, terminate, detected)


if __name__ == "__main__":
    IP = "localhost"
    PORT = 13000

    clientSock = TCP()
    # set non blocking socket
    clientSock.connect(IP, PORT)

    # Logger
    logger = RLLogger()

    # Get starting infor about the subjectid, type of experiment
    ans = clientSock.receive()
    experiment_type = ans['type']
    subjectId = ans['subjectId']
    print("Current experiment: {}".format(experiment_type))

    # If the server starts with Staircase, just wait
    while experiment_type != "UCB":
        ans = clientSock.receive()
        experiment_type = ans['type']
        subjectId = ans['subjectId']

    logger.setType(experiment_type)
    logger.setSubjectId(subjectId)

    # Prepare model's parameters
    discount_factor = 0.98 # not used
    MAX_EXPLORATION_RATE = 1
    MIN_EXPLORATION_RATE = 0.01
    MAX_LEARNING_RATE = 0.5
    MIN_LEARNING_RATE = 0.001
    MAX_TIMES = 100 # 60 works originally for experiment

    # Distortion gains
    values = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 4, 5, 7, 10]

    distortion = Distortion(init_val=1.5, values=values, tcp=clientSock, logger=logger)

    model = UCB(num_states=len(values), max_er=MAX_EXPLORATION_RATE, min_er=MIN_EXPLORATION_RATE,
                discount_factor=discount_factor, max_lr=MAX_LEARNING_RATE, min_lr=MIN_LEARNING_RATE,
                distortion=distortion, MAX_TIMES=MAX_TIMES, subjectId=subjectId)

    times, converged = model.optimize_policy()

    best_distortion = values[np.argmax(model.q_table)]

    # Converged
    # Send to server that the algorithm converged
    clientSock.send(best_distortion, convergence=True)
    logger.logUCB(best_distortion, False, False, convergence=converged)

    print("Best_actions: {}, converged: {}".format(best_distortion, converged))
    print("q_table: \n {}".format(model.q_table))





