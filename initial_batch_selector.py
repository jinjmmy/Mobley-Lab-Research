import pandas as pd
import numpy as np
from numpy.random import choice
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools

class CompoundSelection:
    def __init__(self, descriptors_file, num_picks=25):
        self.descriptors_file = descriptors_file
        self.num_picks = num_picks
        self.descriptors = self.load_descriptors()
        self.compound_names = self.descriptors['Name']
        self.data = self.descriptors.drop('Name', axis=1).to_numpy()

    def load_descriptors(self):
        """Load descriptors from the CSV file."""
        descriptors = pd.read_csv(self.descriptors_file)
        return descriptors

    def perform_tsne(self):
        """Perform t-SNE using the Morgan fingerprints (ECFP) and return the 2D projection."""
        tsne = TSNE(n_components=2, init='random', random_state=1234)
        X_tsne = tsne.fit_transform(self.data)
        return X_tsne

    def plot_tsne(self, X_tsne):
        """Plot the 2D t-SNE projection."""
        plt.figure(figsize=(16, 10))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='b', marker='o', s=15, alpha=0.5)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE using Morgan fingerprints')
        plt.show()

    def rejection_sampling(self, probabilities):
        """Perform rejection sampling to ensure unique selections."""
        unique_indices = set()
        while len(unique_indices) < self.num_picks:
            new_index = np.random.choice(np.arange(len(probabilities)), p=probabilities)
            unique_indices.add(new_index)
        return np.array(list(unique_indices))

    def select_compounds(self, X_tsne):
        """Select compounds based on t-SNE projection and rejection sampling."""
        # Create a 2D histogram to estimate probabilities for selection
        lims2 = [(-130, 130), (-130, 130)]
        bins = [260, 260]
        counts2, edges2 = np.histogramdd(X_tsne, bins=bins, range=lims2, density=False)

        starts2 = np.array([edges2[i][0] for i in range(2)])
        steps2 = np.array([(edges2[i][-1] - edges2[i][0]) / (len(edges2[i]) - 1) for i in range(2)])

        inds2 = np.floor((X_tsne - starts2[np.newaxis, :]) / steps2[np.newaxis, :]).astype(int)
        probabilities2 = 1 / counts2[inds2[:, 0], inds2[:, 1]]
        probabilities2 /= np.sum(probabilities2)

        # Perform rejection sampling
        draw_step_0 = self.rejection_sampling(probabilities2)
        S2_sel = X_tsne[draw_step_0, :]

        # Get compound names for the selected indices
        selected_compound_names = self.compound_names.iloc[draw_step_0]
        return selected_compound_names

    def merge_selected_compounds(self, selected_compound_names):
        """Merge selected compounds with the original descriptors."""
        merged_df = pd.merge(self.descriptors, selected_compound_names, on='Name')
        selected_compound_info = merged_df.drop_duplicates(subset=['Name'])
        return selected_compound_info

    def save_selected_compounds(self, selected_compound_info, output_file):
        """Save the selected compounds to a new CSV file."""
        selected_compound_info.to_csv(output_file, index=False)
        print(f'Selected compounds saved to {output_file}')

    def compare_selected_compounds(self, file1, file2):
        """Compare selected compounds with another file to ensure they are unique."""
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        df1['Name'] = df1['Name'].str.strip().str.lower()
        df2['Name'] = df2['Name'].str.strip().str.lower()

        unique_compounds_df1 = set(df1['Name'])
        unique_compounds_df2 = set(df2['Name'])

        common_compounds = unique_compounds_df1.intersection(unique_compounds_df2)
        print("\nCommon compounds:", common_compounds)


# Example usage:
# if __name__ == "__main__":
#     # Initialize the CompoundSelection class with the descriptors file
#     compound_selector = CompoundSelection('docked_ecfp.csv')

#     # Perform t-SNE and plot the results
#     X_tsne = compound_selector.perform_tsne()
#     compound_selector.plot_tsne(X_tsne)

#     # Select compounds based on t-SNE projection
#     selected_compounds = compound_selector.select_compounds(X_tsne)

#     # Merge the selected compounds with the original descriptors
#     selected_compound_info = compound_selector.merge_selected_compounds(selected_compounds)

#     # Save the selected compounds to a new file
#     compound_selector.save_selected_compounds(selected_compound_info, 'ecfp_euclidean_set/new/round0_25_ecfp.csv')

#     # Compare the selected compounds with another file
#     compound_selector.compare_selected_compounds('ecfp_euclidean_set/new/greedy/weighted_random/1/round0_10_ecfp.csv', 
#                                                  'ecfp_euclidean_set/new/round0_25_ecfp.csv')
