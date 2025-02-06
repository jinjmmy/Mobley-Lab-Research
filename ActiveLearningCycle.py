import pandas as pd
import numpy as np
import os
from collections import defaultdict

from sklearn.gaussian_process import GaussianProcessRegressor
import TanimotoKernel

#from sklearn.gaussian_process import GaussianProcessRegressor

class ActiveLearningCycle:
    def __init__(self, initial_selection,input_csv, descriptors_csv, binders_csv, num_greedy=5, num_uncertain=5, output_dir=""):
        self.iter_selection = initial_selection
        self.input_csv = input_csv 
        self.all_docked = descriptors_csv
        self.binders_csv = binders_csv
        self.num_uncertain = num_uncertain
        self.num_greedy = num_greedy
        self.output_dir = output_dir
        self.round_number = 0
        self.training_file = None
        self.test_file = None
        self.predicted_file = None
        self.fileHistory = defaultdict() #key is round # and file type predicted, test, train
        os.makedirs(output_dir, exist_ok=True)


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
        training_file = self._generate_file_name(f"{strategy}_train_cmpds", "csv")
        trainKey = "training_round_" + str(self.round_number)
        self.fileHistory[trainKey] = training_file

        self.select_scores(self.input_csv,self.iter_selection, training_file)
        self.append_descriptors_to_csv(training_file,self.all_docked)
        
        trained_gpr = self.train_gpr_model(training_file)
        compounds_to_remove = training_file
        test_csv = self._generate_file_name(f"{strategy}_test_cmpds", "csv")
        self.remove_train_compounds(self.all_docked,compounds_to_remove,test_csv)
        predicted_csv = self._generate_file_name(f"{strategy}_predicted_results", "csv")
        self.predict_and_save_results(trained_gpr,self.all_docked, predicted_csv)
        # Pick the next 100 train compounds based on highest uncertainty
        temp = self.round_number + 1
        next_round_csv = os.path.join(self.output_dir, f"{strategy}_round{temp}.csv")
        self.select_next_batch(predicted_csv, test_csv, next_round_csv, 'uncertain', 100)
        recall_value = self.calculate_recall(predicted_csv,self.binders_csv,top_n=2000)
        print(f"Round {self.round_number} recall: {recall_value}")

        #at the end of the round change the iteration file self.iter_selection
        self.iter_selection = next_round_csv


#keep track of round number to dynamically name output files
    def _generate_file_name(self, prefix, extension):
        return os.path.join(self.output_dir, f"{prefix}_round{self.round_number}.{extension}")
    

    def run(self):
        for _ in range(self.num_uncertain):
            #select next batch passing in strategy
            self.run_round("uncertain")
            self.round_number += 1


        for _ in range(self.num_greedy):
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

    def train_gpr_model(self,csv_file_path):
    # Load data from CSV
        data = pd.read_csv(csv_file_path)

        # Extract individual bit columns for the representations needed for X_train
        bit_columns = data.drop(columns=['Name', 'SMILES', 'Score'])

        # Convert bits to NumPy array
        X_train = np.array(bit_columns)

        # Target values which in this case are the docking scores for the training data
        y_train = data['Score']

        # Use the custom kernel in a Gaussian process
        gpr = GaussianProcessRegressor(kernel=TanimotoKernel.TanimotoKernel(), n_restarts_optimizer=100).fit(X_train, y_train)

        return gpr
    
    def remove_train_compounds(self,input_csv_path, compounds_to_remove_csv_path, output_csv_path):
        # Read the main compounds CSV file
        all_compounds_df = pd.read_csv(input_csv_path)

        # Read the list of compounds to remove
        compounds_to_remove_df = pd.read_csv(compounds_to_remove_csv_path)

        # Identify the indices of compounds to remove
        indices_to_remove = all_compounds_df[all_compounds_df['Name'].isin(compounds_to_remove_df['Name'])].index

        # Remove compounds from the main DataFrame
        remaining_compounds_df = all_compounds_df.drop(indices_to_remove)

        # Save the 'Name' and 'SMILES' columns to a new CSV file
        remaining_compounds_df[['Name', 'SMILES']].to_csv(output_csv_path, index=False)

        print(f"Compounds removed and remaining Name and SMILES saved to {output_csv_path}")


    def predict_and_save_results(self, gpr, csv_file, output_csv):
        # Load data from CSV
        data = pd.read_csv(csv_file)

        # Extract individual bit columns for the representations needed for X_test
        bit_columns = data.drop(columns=['Name', 'SMILES'])

        # Convert bits to NumPy array
        X_test = np.array(bit_columns)

        # Predict using the Gaussian process model and obtain covariance
        y_pred, sigma = gpr.predict(X_test, return_std=True)

        # Add predicted values and uncertainty to the DataFrame
        data['Predicted_Score'] = y_pred
        data['Uncertainty'] = sigma

        # Save the DataFrame to a new CSV file
        output_data = data[['Name', 'SMILES', 'Predicted_Score', 'Uncertainty']]
        output_data.to_csv(output_csv, index=False)

    def calculate_recall(self,predictions_file, binders_file, top_n=2000, on_column='Name'):
        # Load the predicted results CSV
        predicted_results = pd.read_csv(predictions_file)

        # Sort the DataFrame based on Predicted_Score in ascending order
        sorted_results = predicted_results.sort_values(by='Predicted_Score', ascending=True)

        # Select the top N compounds
        top_compounds = sorted_results.head(top_n)

        # Read the data from the binders CSV file into a pandas DataFrame
        actual_df = pd.read_csv(binders_file)

        # Extract the compounds (e.g., 'Name') from each DataFrame
        compounds_file = set(actual_df[on_column])
        compounds_predicted = set(top_compounds[on_column])

        # Find the common compounds between the two DataFrames
        common_compounds = compounds_file.intersection(compounds_predicted)

        # Count the number of common compounds
        true_positives_count = len(common_compounds)

        # Count the number of false negatives
        false_negatives_count = len(compounds_file - common_compounds)

        # Calculate recall
        recall = true_positives_count / (true_positives_count + false_negatives_count)

        return recall
