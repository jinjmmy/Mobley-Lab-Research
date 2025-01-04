import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve

class ActiveLearningCycle:
    def __init__(self, train_data, test_data, rounds, greedy_rounds=5, uncertain_rounds=5, save_path=""):
        self.train_data = train_data
        self.test_data = test_data
        self.rounds = rounds
        self.greedy_rounds = greedy_rounds
        self.uncertain_rounds = uncertain_rounds
        self.save_path = save_path

#run greedy rounds first to explore.

#automate the removing of training data from the whole set


    