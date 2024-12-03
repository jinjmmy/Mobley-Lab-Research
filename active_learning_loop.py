class ActiveLearningLoop:
    def __init__(self, active_learning, num_cycles, uncertain_cycles, greedy_cycles):
        self.active_learning = active_learning
        self.num_cycles = num_cycles
        self.uncertain_cycles = uncertain_cycles
        self.greedy_cycles = greedy_cycles
    
    def run(self):
        """Run the active learning process over a number of cycles, alternating strategies."""
        for cycle in range(self.num_cycles):
            if cycle < self.uncertain_cycles:
                selection_strategy = 'uncertain'
            else:
                selection_strategy = 'greedy'
                
            # Run a round with the specified strategy
            self.active_learning.run_round(strategy=selection_strategy)
            
            # Save files for each round
            self.active_learning.save_predicted_results(cycle)
            self.active_learning.save_train_cmpds(cycle)
