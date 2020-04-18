import pandas as pd
from datetime import date
import os
import json

m_subId = 'subjectId'
m_type = 'type'
m_trial = 'trial'
m_gain = 'gain'
m_detected = 'detected'
m_original_detected = 'original_detected'
m_isTurn = 'isTurn'
m_radius = 'radius'
m_convergence = 'converged'
m_dataDict = 'dataDict'


class RLLogger:
    def __init__(self, filename=None):
        """Set up the logger"""
        self.UCBheaders = [m_subId, m_type, m_trial, m_gain, m_original_detected, m_detected, m_convergence]

        if filename is None:
            self.UCBdataDict = dict.fromkeys(self.UCBheaders)
            for head in self.UCBheaders:
                self.UCBdataDict[head] = []

            self.UCBheaders2 = [m_subId, m_type, m_trial, m_radius, m_gain, m_original_detected, m_detected, m_convergence]
            self.UCBdataDict2 = dict.fromkeys(self.UCBheaders2)
            for head in self.UCBheaders2:
                self.UCBdataDict2[head] = []

            self.trial = 0
            self.subject_id = -1
            self.type = "UCB"
            self.dataDict = dict()
        else:
            """Load a already trained model"""
            print("Reading model : {}".format(filename))
            with open(filename, 'r') as infile:
                data = json.load(infile)
                self.trial = data[m_trial]
                self.subject_id = data[m_subId]
                self.type = data[m_type]
                self.dataDict = data[m_dataDict]
                if self.type == "UCB":
                    self.UCBdataDict = self.dataDict
                else:
                    self.UCBdataDict2 = self.dataDict

    def setType(self, type):
        self.type = type
        if type == "UCB":
            self.dataDict = self.UCBdataDict
        else:
            self.dataDict = self.UCBdataDict2

    def setSubjectId(self, subject_id):
        self.subject_id = subject_id

    def logUCB(self, gain, original_detected, detected, convergence=False):
        """Save the UCB model (varying 1 parameter)"""
        self.UCBdataDict[m_subId].append(self.subject_id)
        self.UCBdataDict[m_type].append(self.type)
        self.UCBdataDict[m_trial].append(self.trial)
        self.UCBdataDict[m_gain].append(gain)
        self.UCBdataDict[m_original_detected].append(original_detected)
        self.UCBdataDict[m_detected].append(detected)
        self.UCBdataDict[m_convergence].append(convergence)
        self.trial += 1
        # print("Current data: \n{}".format(pd.DataFrame(data=self.dataDict)))
        self.saveToFile() # save at each line
        self.save_model()

    def logUCB2(self, gain, radius, original_detected, detected, convergence=False):
        """Save the UCB model (varying 2 parameters)"""
        self.UCBdataDict2[m_subId].append(self.subject_id)
        self.UCBdataDict2[m_type].append(self.type)
        self.UCBdataDict2[m_trial].append(self.trial)
        self.UCBdataDict2[m_radius].append(radius)
        self.UCBdataDict2[m_gain].append(gain)
        self.UCBdataDict2[m_original_detected].append(original_detected)
        self.UCBdataDict2[m_detected].append(detected)
        self.UCBdataDict2[m_convergence].append(convergence)
        self.trial += 1
        self.saveToFile() # save at each line
        self.save_model()

    def save_model(self):
        filename = "../Models/Logger_Subject_{}_type_{}_trials_{}_{}".format(self.subject_id, self.type, self.trial, date.today())
        data = dict({
            m_subId : int(self.subject_id),
            m_trial : int(self.trial),
            m_type : str(self.type),
            m_dataDict : self.dataDict
        })

        # Remove old file and save the new one with the new line written
        if self.trial > 1:
            prev_file = "../Models/Logger_Subject_{}_type_{}_trials_{}_{}.json".format(self.subject_id, self.type, self.trial - 1, date.today())
            os.remove(prev_file)

        print("Saving model {}".format(filename))
        with open('{}.json'.format(filename), 'w') as outfile:
            json.dump(data, outfile)

    def saveToFile(self):
        if self.type == "UCB":
            df = pd.DataFrame(data=self.UCBdataDict)
        else:
            df = pd.DataFrame(data=self.UCBdataDict2)
        # Remove old file
        if self.trial > 1:
            try :
                prev_file = "../ExportData/Subject_{}_type_{}_{}_trials_{}.csv".format(self.subject_id, self.type, self.trial - 1, date.today())
                os.remove(prev_file)
            except:
                print("Didnt remove")

        filename: str = "../ExportData/Subject_{}_type_{}_{}_trials_{}.csv".format(self.subject_id, self.type, self.trial, date.today())
        print("Saving {}".format(filename))
        df.to_csv(filename, index=False)






