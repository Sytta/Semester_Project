from Communication.Client import TCP
from Algorithms.UCB_2_param_combined import UCB2
import numpy as np
from Logger.RLLogger import *

class Distortion:
    def __init__(self, init_val, R, Gain, tcp, logger):
        self.distortion = init_val
        self.R_max = R[-1]
        self.G_max = Gain[-1]
        self.R = R
        self.G = Gain

        self.tcp = tcp
        self.logger = logger

        self.values = []
        for r in self.R:
            for g in self.G:
                self.values.append((r, g))

        # Construct rewards count
        self.construct_rewards_count()

    def construct_rewards_count(self):
        # Prepare to keep track of the all the rewards received
        neg_rewards = [(-r, -g) for (r, g) in self.values]
        all_rewards = self.values + neg_rewards
        self.rewards_count = dict.fromkeys(all_rewards, 0)

    def correct_reward(self, reward, terminate, detected):
        # Correct reward
        # Terminate is just a parameter for the UCB algorithm, not really important, you can ignore it
        # Eg: for the past, for gain 5, if the subject never noticed, and this time he detected, we modify
        # his reaction to have not detected as well
        self.rewards_count[reward] += 1
        pos_count = self.rewards_count[(abs(reward[0]), abs(reward[1]))]
        neg_count = self.rewards_count[(-abs(reward[0]), -abs(reward[1]))]

        detected_corrected = detected

        # If he has detected the gain most of the time
        if pos_count > neg_count:
            r = (1, abs(reward[1]))
            detected_corrected = False
        # If he has not detected the gain most of the time
        elif neg_count > pos_count:
            r = (-1, -abs(reward[1]))
            detected_corrected = True
        else:
        # If the number of detection = number of no detection, do not modify the reward
            if detected:
                r = (-1, -abs(reward[1]))
            else:
                r = (1, abs(reward[1]))

        self.logger.logUCB2(gain=abs(reward[0]), radius=abs(reward[1]), original_detected=detected, detected=detected_corrected)

        print("Detected {} - Original Reward: {}, Corrected Reward: {}, Terminate: {}".format(detected, reward, r, terminate))
        return r, terminate

    def step(self, action):
        # UCB calls this function
        # Get the distortion from the subject, correct the reward, and send it to the UCB algorithm

        # Choose the gain, radius depending on the action UCB picked
        (radius, gain) = self.values[action]

        # Send the gain and radius to the subject and get his response
        self.tcp.send2(gain=gain, radius=radius)
        answers = self.tcp.receive()
        gain = answers['distortion']
        radius = answers['radius']
        detected = answers['detected']

        # Send corrected reward to UCB
        if detected:
            return self.correct_reward((-radius, -gain), False, detected)
        else:
            return self.correct_reward((radius, gain), False, detected)


if __name__ == "__main__":
    IP = "localhost"
    PORT = 13000

    clientSock = TCP()
    # # set non blocking socket
    clientSock.connect(IP, PORT)

    # Logger
    logger = RLLogger()

    # Get starting infor about the subjectid, type of experiment
    ans = clientSock.receive()
    experiment_type = ans['type']
    subjectId = ans['subjectId']
    print("Current experiment: {}".format(experiment_type))

    # If the server starts with Staircase, just wait
    while experiment_type != "UCB2":
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
    MAX_TIMES = 300

    # Drange values
    R = np.linspace(2, 16, 8)  # 8 values
    # Distortion gains
    Gain = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 4, 5, 7, 10] # 11 values

    distortion = Distortion((0, 0), R=R, Gain=Gain, tcp=clientSock, logger=logger)

    model = UCB2(num_states_G=len(Gain), num_states_R=len(R), max_er=MAX_EXPLORATION_RATE, min_er=MIN_EXPLORATION_RATE,
                 discount_factor=discount_factor, max_lr=MAX_LEARNING_RATE, min_lr=MIN_LEARNING_RATE,
                 distortion=distortion, er_times=51, lr_times=168, count_times=270, subjectId=subjectId)

    times, converged = model.optimize_policy()

    # Converged or terminated
    max_action_R = distortion.R[np.argmax(model.q_table_R)]
    max_action_G = distortion.G[np.argmax(model.q_table_G)]

    # Log convergence
    logger.logUCB2(radius=max_action_R, gain=max_action_G, detected=False, original_detected=False, convergence=converged)

    # Send to server that the algorithm converged
    clientSock.send2(radius=max_action_R, gain=max_action_G, convergence=True)

    print("Best_actions: R = {}, G = {}, converged: {}".format(max_action_R, max_action_G, converged))
    print("q_table_R: \n {}".format(model.q_table_R))
    print("q_table_G: \n {}".format(model.q_table_G))






