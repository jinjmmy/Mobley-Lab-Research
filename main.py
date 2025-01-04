from ActiveLearningCycle import ActiveLearningCycle
from initial_batch_selector import InitialBatchSelector

def main():
    # Step 1: Initial batch selection
    selector = InitialBatchSelector(input_file='docked_ecfp.csv')
    selector.run_tsne()
    selector.select_initial_batch(num_samples=25)
    selector.save_selected_batch(output_file='initial_batch.csv')

    # Step 2: Active learning loop
    uncertain_cycles = 6
    greedy_cycles = 4
    output_dir = 'results/'

    #random strategy called outside first to get docked, round 0
    #random_strategy()

    #files needed:
    #'7nsw_all_hybrid.csv', docked_ecfp.csv, binders_docking.csv
    al_loop = ActiveLearningCycle(
        '7nsw_all_hybrid.csv',
        '../../../../docked_ecfp.csv',
        '../../../../binders_docking.csv'
        num_uncertain=uncertain_cycles,
        num_greedy=greedy_cycles,
        output_dir=output_dir
    )


if __name__ == '__main__':
    main()
