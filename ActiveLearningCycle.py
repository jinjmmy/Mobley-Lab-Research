import pandas as pd
import numpy as np
import os

class ActiveLearningCycle:
    def __init__(self, input_csv, descriptors_csv, binders_csv, greedy_rounds=5, uncertain_rounds=5, output_directory=""):
        self.input_csv = input_csv 
        self.descriptors_csv = descriptors_csv
        self.binders_csv = binders_csv
        self.uncertain_rounds = uncertain_rounds
        self.greedy_rounds = greedy_rounds
        self.output_directory = output_directory
        self.round_number = 0
        self.training_file = None
        self.test_file = None
        self.predicted_file = None
        os.makedirs(output_directory, exist_ok=True)

#run uncertain rounds first to explore.
    def select_scores(self,inp_file,cmpds_file,train_file):
        pass

#automate the removing of training data from the whole set
    def run_round(self,strategy):
        self.round_number += 1
        # Step 1: Select scores and save training set
        cmpds_file = self._generate_file_name(f"{strategy}_cmpds", "csv")
        train_file = self._generate_file_name("train_cmpds", "csv")
        self.select_scores(self.input_csv, cmpds_file, train_file)
        pass

#keep track of round number to dynamically name output files
    def _generate_file_name(self, prefix, extension):
        return os.path.join(self.output_dir, f"{prefix}_round{self.round_number}.{extension}")
    

    def run(self):
        for _ in range(self.uncertain_rounds):
            self.run_round("uncertain")

        for _ in range(self.greedy_rounds):
            self.run_round("greedy")

    