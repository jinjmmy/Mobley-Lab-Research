from ActiveLearningCycle import ActiveLearningCycle
from initial_batch_selector import InitialBatchSelector
import pandas as pd


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

def main():
    # Step 1: Initial batch selection
    # selector = InitialBatchSelector(input_file='docked_ecfp.csv')
    # selector.run_tsne()
    # selector.select_initial_batch(num_samples=25)
    # selector.save_selected_batch(output_file='initial_batch.csv')

    # # # Step 2: Active learning loop
    uncertain_cycles = 3
    greedy_cycles = 3
    output_dir = 'results/'

    # #random strategy called outside first to get docked, round 0
    # random_strategy()

#just need the output csv from random strategy
    #files needed:
    #'7nsw_all_hybrid.csv', docked_ecfp.csv, binders_docking.csv
    


    al_loop = ActiveLearningCycle(
        'AL_cycle_example/round0_100_ecfp.csv',
        'setup/7nsw_all_hybrid.csv',
        'setup/docked_ecfp.csv',
        'setup/binders_docking.csv',
        num_uncertain=uncertain_cycles,
        num_greedy=greedy_cycles,
        output_dir=output_dir
    )
    al_loop.run()


if __name__ == '__main__':
    main()
