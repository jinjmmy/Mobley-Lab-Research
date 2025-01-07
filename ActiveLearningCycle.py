import pandas as pd
import numpy as np
import os
from collections import defaultdict

class ActiveLearningCycle:
    def __init__(self, initial_selection,input_csv, descriptors_csv, binders_csv, greedy_rounds=5, uncertain_rounds=5, output_directory=""):
        self.iter_selection = initial_selection
        self.input_csv = input_csv 
        self.all_docked = descriptors_csv
        self.binders_csv = binders_csv
        self.uncertain_rounds = uncertain_rounds
        self.greedy_rounds = greedy_rounds
        self.output_directory = output_directory
        self.round_number = 0
        self.training_file = None
        self.test_file = None
        self.predicted_file = None
        self.fileHistory = defaultdict() #key is round # and file type predicted, test, train
        os.makedirs(output_directory, exist_ok=True)


#run uncertain rounds first to explore.
    def select_scores(self,inp_file,cmpds_file,output_csv_path):
            # Load the main CSV file into a DataFrame
        all_compounds_df = pd.read_csv(inp_file)
        # Load the compounds CSV file into a DataFrame
        cmpds_df = pd.read_csv(cmpds_file)
        # Merge the two DataFrames based on the 'Name' column using an inner join
        merged_df = pd.merge(all_compounds_df, cmpds_df, how='inner', on='Name')
        # Print the columns of the merged DataFrame for inspection
        print("Columns of merged DataFrame:")
        print(merged_df.columns)
        # Select the desired columns
        selected_df = merged_df[['Name', 'SMILES_x', 'Score']]
        # Print the selected DataFrame for further inspection
        print("\nSelected DataFrame:")
        print(selected_df)
        # Rename 'SMILES_x' to 'SMILES'
        selected_df = selected_df.rename(columns={'SMILES_x': 'SMILES'})
        # Save the resulting DataFrame to a new CSV file
        #output csv path will be the training file?
        selected_df.to_csv(output_csv_path, index=False)

        print(f"Selected compounds saved to {output_csv_path}")
        
    def append_descriptors_to_csv(self,compounds_csv_path, descriptors_csv_path):
        # Read the main compounds CSV file
        compounds_df = pd.read_csv(compounds_csv_path)

        # Read the descriptors CSV file
        descriptors_df = pd.read_csv(descriptors_csv_path)

        # Merge the two DataFrames based on the 'Name' column
        merged_df = pd.merge(compounds_df, descriptors_df, on='Name', how='left', suffixes=('', '_descriptor'))

        # Append descriptor columns after the 'Score' column
        score_index = merged_df.columns.get_loc('Score')
        descriptor_columns = [col for col in merged_df.columns if col.endswith('_descriptor')]
        columns_order = list(merged_df.columns[:score_index + 1]) + descriptor_columns + list(merged_df.columns[score_index + 1:])

        # Update DataFrame with the new column order
        merged_df = merged_df[columns_order]

        # Drop the descriptor columns
        merged_df = merged_df.drop(merged_df.filter(like='_descriptor').columns, axis=1)

        # Save the updated DataFrame to the same CSV file
        merged_df.to_csv(compounds_csv_path, index=False)

        print(f"Descriptors appended to {compounds_csv_path}")

#automate the removing of training data from the whole set
    def run_round(self,strategy):
        #self.round_number += 1
        training_file = self._generate_file_name(f"{strategy}_cmpds", "csv")
        self.fileHistory["training_round_"+ self.round_number] = training_file

        self.select_scores(self.input_csv,self.iter_selection, training_file)
        self.append_descriptors_to_csv(training_file,self.all_docked)
        if strategy == 'uncertain':

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
            #select next batch passing in strategy
            self.run_round("uncertain")
            self.round_number += 1

        for _ in range(self.greedy_rounds):
            #select next batch passing in strategy
            self.run_round("greedy")
            self.round_number += 1

    
#every round pass a new csv file into gpr model

    def select_next_batch(self,file1, file2, output_csv, strategy, top_n=None):
        """
        Merges two CSV files based on shared compounds and selects top compounds using the specified strategy.

        Parameters:
        - file1 (str): Path to the first CSV file.
        - file2 (str): Path to the second CSV file.
        - output_csv (str): Path to the output CSV file to save the selected compounds.
        - strategy (str): Strategy for selecting compounds ('uncertain' or 'greedy').
        - top_n (int or None): Number of top compounds to select. If None, selects all (default is None).
        """
        # Load the two CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Merge based on shared compounds
        merged_df = pd.merge(df1, df2, how='inner', on='SMILES')

        # Drop 'Name_y' column
        merged_df.drop(columns=['Name_y'], inplace=True)

        # Rename 'Name_x' to 'Name'
        merged_df.rename(columns={'Name_x': 'Name'}, inplace=True)

        # Determine the strategy for sorting
        if strategy == 'uncertain':
            sorted_results = merged_df.sort_values(by='Uncertainty', ascending=False)
        elif strategy == 'greedy':
            sorted_results = merged_df.sort_values(by='Predicted_Score', ascending=True)
        else:
            raise ValueError("Invalid strategy. Use 'uncertain' or 'greedy'.")

        # Select the top N compounds, or all if top_n is None
        if top_n is not None:
            top_compounds = sorted_results.head(top_n)
        else:
            top_compounds = sorted_results

        # Save the selected compounds to a new CSV file
        top_compounds.to_csv(output_csv, index=False)