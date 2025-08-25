"""Main script to run simulations in parallel or single mode."""


import shutil  
import config
import time
import importlib
import sys
import constants

if config.SCENARIO_NAME == "Base":
    shutil.copy('config.py', 'constants.py')
    shutil.copy('inputs/terminal_data_base.csv',  'inputs/terminal_data.csv')
    import constants
    importlib.reload(constants)

elif config.SCENARIO_NAME == "BottleneckDetection" or config.SCENARIO_NAME == "BreakpointAnalysis":
    if 'constants' in sys.modules:
        del sys.modules['constants']
    import constants

else:
    raise ValueError("Invalid SCENARIO_NAME. It should be either 'Base' or 'BottleneckDetection'.")

import time
import multiprocessing as mp
from tqdm import tqdm
from simulation_analysis.collate_results import collate_results
from simulation_handler.run_simulation import run_simulation
from simulation_handler.helpers import clean_results_directory


def parallel_run():
    """Main function to execute the simulation and collate results."""
    start_time = time.time()
    num_runs = constants.NUM_RUNS
    num_cores = constants.NUM_CORES
    start_seed = constants.START_SEED

    if constants.DELETE_RESULTS_FOLDER:
        clean_results_directory()
    seeds = range(start_seed, start_seed + num_runs)
    with mp.Pool(processes=num_cores) as pool:
        for _ in tqdm(pool.imap_unordered(run_simulation, seeds), total=num_runs, desc="Simulations Progress"):
            pass
    total_time = round((time.time() - start_time) / 60, 2)
    collate_results(num_runs, total_time)


def single_run():
    """Run a single seed simulation."""
    start_time = time.time()
    seed = constants.START_SEED
    if constants.DELETE_RESULTS_FOLDER:
        clean_results_directory()
    run_simulation(seed)
    # collate_results(1)
    total_time = round((time.time() - start_time) / 60, 2)
    print(f"\nTotal time taken: {total_time} minutes")
   
if __name__ == "__main__":
    # single_run() # Comment this line and uncomment the next line to run multiple simulations
    parallel_run()  # Comment this line and uncomment the previous line to run a single simulation

    # if the SCENARIO_NAME is not "Base" or "BreakpointAnalysis", then the results will be saved in a new collated folder
    if constants.SCENARIO_NAME == "BottleneckDetection":
        num_runs = constants.NUM_RUNS
        log_num = constants.LOG_NUM
        ARRIVAL_INCREASE_FACTOR_CTR = constants.ARRIVAL_INCREASE_FACTOR_CTR
        logfileid = f"{int(constants.NUM_MONTHS)}_months_{num_runs}_runs_run_{log_num}"
        collated_folder = f"collatedResults/{logfileid}"
        new_folder_name = f"bottleneckAnalysis/S_{constants.SCENARIO_NAME}_{ARRIVAL_INCREASE_FACTOR_CTR}x_{logfileid}"
        shutil.copytree(collated_folder, new_folder_name)
        shutil.rmtree(collated_folder)
        print(f"Collated results saved in {new_folder_name} folder.")
        
