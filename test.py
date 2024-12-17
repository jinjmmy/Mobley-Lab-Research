# import packages
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from rdkit import DataStructs, Chem
from rdkit.Chem import MolFromSmiles, AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

def random_strategy(docked_file, output_csv, subset_size=None, random_seed=None):
    """
    Randomly selects a subset of compounds from the docking results CSV.

    Parameters:
    - docked_file (str): Path to the CSV file with docking results.
    - output_csv (str): Path to the output CSV file to save the selected compounds.
    - subset_size (int or None): Number of compounds to randomly select. If None, selects all (default is None).
    - random_seed (int or None): Seed for reproducibility in random sampling. If None, no seed is set (default is None).
    """
    # Load the docking results CSV
    docking_results = pd.read_csv(docked_file)

    # Randomly select a subset of compounds
    if subset_size is not None:
        selected_compounds = docking_results.sample(n=subset_size, random_state=random_seed)
    else:
        selected_compounds = docking_results

    # Save the selected compounds to a new CSV file
    selected_compounds.to_csv(output_csv, index=False)
def select_next_batch(file1, file2, output_csv, strategy, top_n=None):
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
def uncertain_strategy(predictions_file, output_csv, top_n=None):
    """
    Sorts the predicted results based on uncertainty and selects the top N compounds.

    Parameters:
    - predictions_file (str): Path to the CSV file with predicted results.
    - output_csv (str): Path to the output CSV file to save the selected compounds.
    - top_n (int or None): Number of top compounds to select. If None, selects all (default is None).
    """
    # Load the predicted results CSV
    predicted_results = pd.read_csv(predictions_file)

    # Sort the DataFrame based on uncertainty in descending order
    sorted_results = predicted_results.sort_values(by='Uncertainty', ascending=False)

    # Select the top N compounds, or all if top_n is None
    if top_n is not None:
        top_compounds = sorted_results.head(top_n)
    else:
        top_compounds = sorted_results

    # Save the selected compounds to a new CSV file
    top_compounds.to_csv(output_csv, index=False)
def greedy_strategy(predictions_file, output_csv, top_n=None):
    """
    Sorts the predicted results based on greedy and selects the top N compounds.

    Parameters:
    - predictions_file (str): Path to the CSV file with predicted results.
    - output_csv (str): Path to the output CSV file to save the selected compounds.
    - top_n (int or None): Number of top compounds to select. If None, selects all (default is None).
    """
    # Load the predicted results CSV
    predicted_results = pd.read_csv(predictions_file)

    # Sort the DataFrame based on greedy in descending order
    sorted_results = predicted_results.sort_values(by='Predicted_Score', ascending=True)

    # Select the top N compounds, or all if top_n is None
    if top_n is not None:
        top_compounds = sorted_results.head(top_n)
    else:
        top_compounds = sorted_results

    # Save the selected compounds to a new CSV file
    top_compounds.to_csv(output_csv, index=False)
def select_scores(input_csv_path, cmpds_csv_path, output_csv_path):
    # Load the main CSV file into a DataFrame
    all_compounds_df = pd.read_csv(input_csv_path)

    # Load the compounds CSV file into a DataFrame
    cmpds_df = pd.read_csv(cmpds_csv_path)

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
    selected_df.to_csv(output_csv_path, index=False)

    print(f"Selected compounds saved to {output_csv_path}")
def append_descriptors_to_csv(compounds_csv_path, descriptors_csv_path):
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
"""Set up Gaussian Process Regressor model"""
# Function to calculate the tanimoto similarity for the Gaussian process kernel prediction
def tanimoto_similarity(a, b):
    """Computes the Tanimoto similarity for all pairs.

  Args:
    a: Numpy array with shape [batch_size_a, num_features].
    b: Numpy array with shape [batch_size_b, num_features].

  Returns:
    Numpy array with shape [batch_size_a, batch_size_b].
  """
    aa = np.sum(a, axis=1, keepdims=True)
    bb = np.sum(b, axis=1, keepdims=True)
    ab = np.matmul(a, b.T)
    return np.true_divide(ab, aa + bb.T - ab)
class TanimotoKernel(gaussian_process.kernels.NormalizedKernelMixin,
                     gaussian_process.kernels.StationaryKernelMixin,
                     gaussian_process.kernels.Kernel):
  """Custom Gaussian process kernel that computes Tanimoto similarity."""

  def __init__(self):
    """Initializer."""
    pass  # Does nothing; this is required by get_params().

  def __call__(self, X, Y=None, eval_gradient=False):  # pylint: disable=invalid-name
    """Computes the pairwise Tanimoto similarity.

    Args:
      X: Numpy array with shape [batch_size_a, num_features].
      Y: Numpy array with shape [batch_size_b, num_features]. If None, X is
        used.
      eval_gradient: Whether to compute the gradient.

    Returns:
      Numpy array with shape [batch_size_a, batch_size_b].

    Raises:
      NotImplementedError: If eval_gradient is True.
    """
    if eval_gradient:
      raise NotImplementedError
    if Y is None:
      Y = X
    return tanimoto_similarity(X, Y)
def train_gpr_model(csv_file_path):
    # Load data from CSV
    data = pd.read_csv(csv_file_path)

    # Extract individual bit columns for the representations needed for X_train
    bit_columns = data.drop(columns=['Name', 'SMILES', 'Score'])

    # Convert bits to NumPy array
    X_train = np.array(bit_columns)

    # Target values which in this case are the docking scores for the training data
    y_train = data['Score']

    # Use the custom kernel in a Gaussian process
    gpr = GaussianProcessRegressor(kernel=TanimotoKernel(), n_restarts_optimizer=100).fit(X_train, y_train)

    return gpr
def remove_train_compounds(input_csv_path, compounds_to_remove_csv_path, output_csv_path):
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
def predict_and_save_results(gpr, csv_file, output_csv):
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
def append_train_compounds(file1_path, file2_path, output_path):
    """
    Concatenates two CSV files with the same format and saves the result to a new CSV file.

    Parameters:
    - file1_path (str): Path to the first CSV file.
    - file2_path (str): Path to the second CSV file.
    - output_path (str): Path to save the concatenated CSV file.
    """
    # Load the data from both CSV files
    data1 = pd.read_csv(file1_path)
    data2 = pd.read_csv(file2_path)

    # Concatenate the two DataFrames
    concatenated_data = pd.concat([data1, data2], ignore_index=True)

    # Save the concatenated DataFrame to a new CSV file
    concatenated_data.to_csv(output_path, index=False)

    print(f"Concatenated compounds saved to {output_path}")
def calculate_recall(predictions_file, binders_file, top_n=2000, on_column='Name'):
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
_________________________________________________________________________________________________________
# Choose a random subset of compounds to start with.
docked_file_path = '../../../docked_ecfp.csv'
output_csv_path = 'round0_1000_ecfp.csv'
subset_size_to_select = 100  # Set the desired subset size

# Call the function
random_strategy(docked_file_path, output_csv_path, subset_size=subset_size_to_select)

# Unblind the scores for these 100 compounds and save as training set.
input_csv_path = '../../../../7nsw_all_hybrid.csv'
cmpds_csv_path = 'round0_100_ecfp.csv'
output_csv_path = 'round0_100_train_cmpds.csv'

select_scores(input_csv_path, cmpds_csv_path, output_csv_path)
# Extract the descriptors and append them to the training set.
compounds_csv_path = 'round0_100_train_cmpds.csv'
descriptors_csv_path = '../../../../docked_ecfp.csv'

append_descriptors_to_csv(compounds_csv_path, descriptors_csv_path)
# Train our model using this training file.
csv_file_path = 'round0_100_train_cmpds.csv'
trained_gpr = train_gpr_model(csv_file_path)
# Remove the training set from the test set
# Save the remainder of compounds as next rounds test set
input_csv_path = '../../../../docked_ecfp.csv'
compounds_to_remove_csv_path = 'round0_100_train_cmpds.csv'
output_csv_path = 'round0_100_test_cmpds.csv'

remove_train_compounds(input_csv_path, compounds_to_remove_csv_path, output_csv_path)
# Predict on the entire library
csv_file_to_predict = '../../../../docked_ecfp.csv'
output_csv_file = 'round0_100_predicted_results.csv'
predict_and_save_results(trained_gpr, csv_file_to_predict, output_csv_file)
# Pick the next 100 train compounds based on highest uncertainty
file1 = 'round0_100_predicted_results.csv'
file2 = 'round0_100_test_cmpds.csv'
output_csv = 'round1_100_cmpds.csv'
strategy = 'uncertain'  # or 'greedy'
top_n = 100  # or None for all

select_next_batch(file1, file2, output_csv, strategy, top_n)
# Calculate recall for round 1
predictions_file_path = 'round0_100_predicted_results.csv'
binders_file_path = '../../../../binders_docking.csv'
# Calculate Recall
recall_value = calculate_recall(predictions_file_path, binders_file_path, top_n=2000)
print("Recall:", recall_value)




"""ROUND 1"""
# Unblind the scores for these 100 compounds and save as training set.
input_csv_path = '../../../../7nsw_all_hybrid.csv'
cmpds_csv_path = 'round1_100_cmpds.csv'
output_csv_path = 'round1_100_train_cmpds.csv'

select_scores(input_csv_path, cmpds_csv_path, output_csv_path)
# Extract the descriptors and append them to the training set.
compounds_csv_path = 'round1_100_train_cmpds.csv'
descriptors_csv_path = '../../../../docked_ecfp.csv'

append_descriptors_to_csv(compounds_csv_path, descriptors_csv_path)
# Concatanate the train sets together 
file1_path = 'round0_100_train_cmpds.csv'
file2_path = 'round1_100_train_cmpds.csv'
output_path = 'round1_100_train_cmpds.csv'

append_train_compounds(file1_path, file2_path, output_path)
# Retrain the model using this training file.
csv_file_path = 'round1_100_train_cmpds.csv'
trained_gpr = train_gpr_model(csv_file_path)
# Remove the training set from the test set
# Save the remainder of compounds as next rounds test set
input_csv_path = 'round0_100_test_cmpds.csv'
compounds_to_remove_csv_path = 'round1_100_train_cmpds.csv'
output_csv_path = 'round1_100_test_cmpds.csv'

remove_train_compounds(input_csv_path, compounds_to_remove_csv_path, output_csv_path)
# Predict on the entire library
csv_file_to_predict = '../../../../docked_ecfp.csv'
output_csv_file = 'round1_100_predicted_results.csv'
predict_and_save_results(trained_gpr, csv_file_to_predict, output_csv_file)
# Pick the next 100 train compounds based on highest uncertainty
file1 = 'round1_100_predicted_results.csv'
file2 = 'round1_100_test_cmpds.csv'
output_csv = 'round2_100_cmpds.csv'
strategy = 'uncertain'  # or 'greedy'
top_n = 100  # or None for all

select_next_batch(file1, file2, output_csv, strategy, top_n)
# Calculate recall for round 2
predictions_file_path = 'round1_100_predicted_results.csv'
binders_file_path = '../../../../binders_docking.csv'
# Calculate Recall
recall_value = calculate_recall(predictions_file_path, binders_file_path, top_n=2000)
print("Recall:", recall_value)



"""ROUND 2"""
# Unblind the scores for these 100 compounds and save as training set.
input_csv_path = '../../../../7nsw_all_hybrid.csv'
cmpds_csv_path = 'round2_100_cmpds.csv'
output_csv_path = 'round2_100_train_cmpds.csv'

select_scores(input_csv_path, cmpds_csv_path, output_csv_path)
# Extract the descriptors and append them to the training set.
compounds_csv_path = 'round2_100_train_cmpds.csv'
descriptors_csv_path = '../../../../docked_ecfp.csv'

append_descriptors_to_csv(compounds_csv_path, descriptors_csv_path)
# Concatanate the train sets together 
file1_path = 'round1_100_train_cmpds.csv'
file2_path = 'round2_100_train_cmpds.csv'
output_path = 'round2_100_train_cmpds.csv'

append_train_compounds(file1_path, file2_path, output_path)
# Retrain the model using this training file.
csv_file_path = 'round2_100_train_cmpds.csv'
trained_gpr = train_gpr_model(csv_file_path)
# Remove the training set from the test set
# Save the remainder of compounds as next rounds test set
input_csv_path = 'round1_100_test_cmpds.csv'
compounds_to_remove_csv_path = 'round2_100_train_cmpds.csv'
output_csv_path = 'round2_100_test_cmpds.csv'

remove_train_compounds(input_csv_path, compounds_to_remove_csv_path, output_csv_path)
# Predict on the entire library
csv_file_to_predict = '../../../../docked_ecfp.csv'
output_csv_file = 'round2_100_predicted_results.csv'
predict_and_save_results(trained_gpr, csv_file_to_predict, output_csv_file)
# Pick the next 100 train compounds based on highest uncertainty
file1 = 'round2_100_predicted_results.csv'
file2 = 'round2_100_test_cmpds.csv'
output_csv = 'round3_100_cmpds.csv'
strategy = 'uncertain'  # or 'greedy'
top_n = 100  # or None for all

select_next_batch(file1, file2, output_csv, strategy, top_n)
# Calculate recall for round 3
predictions_file_path = 'round2_100_predicted_results.csv'
binders_file_path = '../../../../binders_docking.csv'
# Calculate Recall
recall_value = calculate_recall(predictions_file_path, binders_file_path, top_n=2000)
print("Recall:", recall_value)



