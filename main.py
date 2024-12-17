from ActiveLearningCycle import ActiveLearningCycle
from initial_batch_selector import InitialBatchSelector

def main():
    # Step 1: Initial batch selection
    selector = InitialBatchSelector(input_file='docked_ecfp.csv')
    selector.run_tsne()
    selector.select_initial_batch(num_samples=25)
    selector.save_selected_batch(output_file='initial_batch.csv')

    # Step 2: Active learning loop
    total_cycles = 10
    uncertain_cycles = 6
    greedy_cycles = 4
    output_dir = 'results/'

    al_loop = ActiveLearningCycle(
        initial_training_file='initial_batch.csv',
        num_cycles=total_cycles,
        num_uncertain=uncertain_cycles,
        num_greedy=greedy_cycles,
        output_dir=output_dir
    )
    al_loop.run()

if __name__ == '__main__':
    main()
