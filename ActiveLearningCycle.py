import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve

class ActiveLearningCycle:
    def __init__(self, model, train_data, test_data, rounds, greedy_fraction=0.5, uncertain_fraction=0.5, save_path=""):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.rounds = rounds
        self.greedy_fraction = greedy_fraction
        self.uncertain_fraction = uncertain_fraction
        self.save_path = save_path
        self.history = []  #result history

    def run_round(self, round_num):
        """
        Run a single round of active learning.
        """
        print(f"Running round {round_num}...")
        
        # Train the model on the current training data
        self.model.fit(self.train_data['features'], self.train_data['labels'])
        
        # Get predictions and calculate uncertainty
        predictions = self.model.predict(self.test_data['features'])
        uncertainty = self.calculate_uncertainty(self.test_data)
        
        # Get the greedy and uncertain samples to label
        greedy_samples, uncertain_samples = self.select_samples(uncertainty)
        
        # Add the newly labeled samples to the training data
        self.train_data = self.update_train_data(greedy_samples, uncertain_samples)

        # Save the model's predictions and training data for the current round
        self.save_round_data(round_num, predictions, self.train_data)

        # Evaluate model performance (recall, enrichment)
        recall, enrichment = self.evaluate_model(predictions, self.test_data['labels'])
        self.history.append((round_num, recall, enrichment))

    def select_samples(self, uncertainty):
        """
        Select greedy and uncertain samples based on uncertainty scores.
        """
        num_greedy = int(self.greedy_fraction * len(uncertainty))
        num_uncertain = int(self.uncertain_fraction * len(uncertainty))
        
        # Sort by uncertainty (high to low)
        sorted_indices = np.argsort(uncertainty)
        
        # Select greedy and uncertain samples
        greedy_samples = sorted_indices[:num_greedy]
        uncertain_samples = sorted_indices[num_greedy:num_greedy+num_uncertain]
        
        return greedy_samples, uncertain_samples

    def update_train_data(self, greedy_samples, uncertain_samples):
        """
        Update training data by adding the newly labeled samples.
        """
        new_train_data = self.test_data.iloc[greedy_samples].append(self.test_data.iloc[uncertain_samples])
        self.train_data = self.train_data.append(new_train_data, ignore_index=True)
        return self.train_data

    def calculate_uncertainty(self, data):
        """
        Calculate uncertainty for each sample in the test data.
        """
        # Placeholder: Implement model-specific uncertainty calculation (e.g., entropy, margin sampling)
        predictions = self.model.predict_proba(data['features'])
        uncertainty = -np.max(predictions * np.log(predictions), axis=1)  # Example: entropy-based uncertainty
        return uncertainty

    def evaluate_model(self, predictions, true_labels):
        """
        Evaluate the model's recall and enrichment for this round.
        """
        precision, recall, _ = precision_recall_curve(true_labels, predictions)
        # Placeholder for enrichment calculation
        enrichment = np.sum(recall > 0.8)  # Just a sample metric, adjust as needed
        return recall[-1], enrichment

    def save_round_data(self, round_num, predictions, train_data):
        """
        Save predictions and training data to CSV after each round.
        """
        predicted_results = pd.DataFrame(predictions, columns=['predictions'])
        train_cmpds = train_data.copy()

        # Save the CSV files for this round
        predicted_results.to_csv(f"{self.save_path}/round_{round_num}_predicted_results.csv", index=False)
        train_cmpds.to_csv(f"{self.save_path}/round_{round_num}_train_cmpds.csv", index=False)

    def run_all_rounds(self):
        """
        Run the full set of active learning rounds.
        """
        for round_num in range(1, self.rounds + 1):
            self.run_round(round_num)

# Example usage:
# Define your model, train_data, and test_data as appropriate
# model = YourModel()
# train_data = pd.DataFrame({'features': ..., 'labels': ...})
# test_data = pd.DataFrame({'features': ..., 'labels': ...})

# active_learning = ActiveLearningCycle(model, train_data, test_data, rounds=10, greedy_fraction=0.6, uncertain_fraction=0.4, save_path="path/to/save")
# active_learning.run_all_rounds()
