import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve

class ActiveLearningCycle:
    def __init__(self, input_csv, descriptors_csv, binders_csv, greedy_rounds=5, uncertain_rounds=5, output_directory=""):
        self.input_csv = input_csv 
        self.descriptors_csv = descriptors_csv
        self.binders_csv = binders_csv
        self.uncertain_rounds = uncertain_rounds
        self.greedy_rounds = greedy_rounds
        self.output_directory = output_directory

#run uncertain rounds first to explore.

#automate the removing of training data from the whole set

#keep track of round number to dynamically name output files

    