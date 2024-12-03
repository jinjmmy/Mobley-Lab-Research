import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from numpy.random import choice

class InitialBatchSelector:
    def __init__(self, descriptors_file):
        self.descriptors_file = descriptors_file
        self.compound_names = None
        self.data = None
        self.load_data()

    def load_data(self):
        """Loads the descriptors data from CSV."""
        descriptors = pd.read_csv(self.descriptors_file)
        column_to_exclude = 'SMILES'
        
        if column_to_exclude in descriptors.columns:
            descriptors = descriptors.drop(column_to_exclude, axis=1)
        
        self.compound_names = descriptors['Name']
        descriptors = descriptors.drop('Name', axis=1)
        self.data = descriptors.to_numpy()

    def perform_tsne(self):
        """Performs t-SNE dimensionality reduction on the descriptors."""
        tsne = TSNE(n_components=2, init='random', random_state=1234)
        X_tsne = tsne.fit_transform(self.data)
        return X_tsne

    def select_batch(self, n_picks=25):
        """Selects a batch of compounds using rejection sampling."""
        X_tsne = self.perform_tsne()
        
        lims2 = [(-130, 130), (-130, 130)]
        bins = [260, 260]
        
        counts2, edges2 = np.histogramdd(X_tsne, bins=bins, range=lims2, density=False)
        starts2 = np.array([edges2[i][0] for i in range(2)])
        steps2 = np.array([(edges2[i][-1] - edges2[i][0]) / (len(edges2[i]) - 1) for i in range(2)])
        inds2 = np.floor((X_tsne - starts2[np.newaxis, :]) / steps2[np.newaxis, :]).astype(int)
        
        probabilities2 = 1 / counts2[inds2[:, 0], inds2[:, 1]]
        probabilities2 /= np.sum(probabilities2)
        
        unique_indices = set()
        while len(unique_indices) < n_picks:
            new_index = np.random.choice(np.arange(len(probabilities2)), p=probabilities2)
            unique_indices.add(new_index)
        
        draw_step_0 = np.array(list(unique_indices))
        selected_compound_names = self.compound_names.iloc[draw_step_0]
        
        # Ensure no duplicates
        duplicates_exist = selected_compound_names.duplicated().any()
        if duplicates_exist:
            print("Duplicates exist in selected_compound_names.")
        else:
            print("No duplicates found in selected_compound_names.")
        
        return selected_compound_names

    def save_selected_batch(self, selected_compound_names, output_file):
        """Saves the selected batch to a new CSV file."""
        original_descriptors = pd.read_csv(self.descriptors_file)
        merged_df = pd.merge(original_descriptors, selected_compound_names, on='Name')
        selected_compound_info = merged_df.drop_duplicates(subset=['Name'])
        selected_compound_info.to_csv(output_file, index=False)
        print(f"Selected batch saved to {output_file}")
