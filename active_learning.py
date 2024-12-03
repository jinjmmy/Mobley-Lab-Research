import pandas as pd
import numpy as np

class ActiveLearning:
    def __init__(self, training_data):
        self.training_data = training_data
        self.predicted_results = None
        self.train_cmpds = None

    def run_round(self, strategy='uncertain'):
        """Run a round of active learning with a specific strategy."""
        if strategy == 'uncertain':
            self.predicted_results = self.select_uncertain()
        elif strategy == 'greedy':
            self.predicted_results = self.select_greedy()

        # Update training data and track compounds for each round
        self.train_cmpds = self.update_training_set()

    def select_uncertain(self):
        """Logic for uncertain compound selection."""
        # Placeholder for the selection method
        uncertain_selection = pd.DataFrame({
            'Compound': np.random.choice(self.training_data['Compound'], size=10),
            'Uncertain_Score': np.random.rand(10)
        })
        return uncertain_selection

    def select_greedy(self):
        """Logic for greedy compound selection."""
        # Placeholder for the greedy selection method
        greedy_selection = pd.DataFrame({
            'Compound': np.random.choice(self.training_data['Compound'], size=10),
            'Greedy_Score': np.random.rand(10)
        })
        return greedy_selection

    def update_training_set(self):
        """Update training data after each round."""
        updated_data = self.training_data.sample(frac=0.9)  # Placeholder logic
        return updated_data

    def save_predicted_results(self, cycle):
        """Save predicted results after each round."""
        self.predicted_results.to_csv(f'predicted_results_round{cycle}.csv', index=False)

    def save_train_cmpds(self, cycle):
        """Save training compounds after each round."""
        self.train_cmpds.to_csv(f'train_cmpds_round{cycle}.csv', index=False)
