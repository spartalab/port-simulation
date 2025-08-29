"""
This script collates results from multiple simulation runs, generates plots, and calculates statistics.
TODO: Break down into smaller functions for better readability and maintainability.
"""
from runpy import run_path
import time
import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# #import constants

# Get the parent directory path
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # Add parent directory to sys.path
# sys.path.append(parent_dir)

# # Now you can import your script/module
import constants

from simulation_analysis.capacity import calculate_mean_wait_time, calculate_ultimate_capacity
from simulation_analysis.results import get_dists
import config 
from constants import WARMUP_ITERS, SCENARIO_NAME, base_arrival_rate, ARRIVAL_INCREASE_FACTOR


def collate_results(num_runs, total_time):
    """
    Collates results from multiple simulation runs and generates various plots and statistics.
    The function creates directories for storing results, processes data from each run, 
    and generates plots for wait times, queue lengths, and channel utilization.
    It also calculates mean wait times, standard deviations, and other statistics for different ship types.
    The results are saved in a folder structure under 'collatedResults'.

    Args:
        num_runs (int): Number of simulation runs to collate.
        total_time (int): Total time for the simulation in hours.
    Returns:

    """
    start_time = time.time()

    try:
        log_num = constants.LOG_NUM
    except:
        log_num = "TEST"

    delete_everything = constants.DELETE_EVERYTHING

    logfileid = f"{int(constants.NUM_MONTHS)}_months_{num_runs}_runs_run_{log_num}"

    if delete_everything:
        os.system('rm -rf collatedResults')

    # remove previous results
    os.makedirs('collatedResults', exist_ok=True)
    os.makedirs(f'collatedResults/{logfileid}/', exist_ok=True)
    os.makedirs(f'collatedResults/{logfileid}/data', exist_ok=True)
    os.makedirs(f'collatedResults/{logfileid}/timePlots', exist_ok=True)
    os.makedirs(f'collatedResults/{logfileid}/queuePlots', exist_ok=True)
    os.makedirs(f'collatedResults/{logfileid}/distPlots', exist_ok=True)

    logfilename = f'collatedResults/{logfileid}/run_details_{logfileid}.txt'

    print("Run details:")
    print(f"Number of runs: {num_runs}")
    print(f"Simulation time: {constants.NUM_MONTHS} months")
    with open(f'collatedResults/{logfileid}/run_details_{logfileid}.txt', 'w') as f:
        f.write("Run details:\n")

        f.write(f"Simulation time: {constants.NUM_MONTHS} months\n")
        f.write(f"Number of runs: {num_runs}\n")
        f.write(f"Start seed: {constants.START_SEED}\n\n")
        f.write(f"Expanded channel: {constants.EXPANDED_CHANNEL}\n")
        f.write(f"Hurricane scenario: {constants.MODEL_HURRICANE}\n")
        f.write(f"Fog scenario: {constants.MODEL_FOG}\n")
        f.write("\nConstants:\n")
        f.write(
            f"Arrival increase factor: {constants.ARRIVAL_INCREASE_FACTOR}\n")
        f.write(
            f"Anchorage waiting of container: {constants.ANCHORAGE_WAITING_CONTAINER}\n")
        f.write(
            f"Anchorage waiting of liquid: {constants.ANCHORAGE_WAITING_LIQUID}\n")
        f.write(
            f"Anchorage waiting of dry bulk: {constants.ANCHORAGE_WAITING_DRYBULK}\n")
        f.write(
            f"Efficiency of container terminal: {constants.CTR_TERMINAL_EFFICIENCY}\n")
        f.write(
            f"Efficiency of liquid terminal: {constants.LIQ_TERMINAL_EFFICIENCY}\n")
        f.write(
            f"Efficiency of dry bulk terminal: {constants.DRYBULK_TERMINAL_EFFICIENCY}\n")
        f.write(f"Channel safety two way: {constants.CHANNEL_SAFETWOWAY}\n")
        f.write("Arrival rates:\n")
        f.write(f"Container: {constants.mean_interarrival_time_container}\n")
        f.write(f"Liquid: {constants.mean_interarrival_time_tanker}\n")
        f.write(f"Dry Bulk: {constants.mean_interarrival_time_gencargo}\n")

    # distribution plots for each terminal
    for terminal in ['Container', 'Liquid', 'Dry Bulk']:

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, num_runs))
        means_for_all_seeds = []
        stds_for_all_seeds = []

        for seed in range(constants.START_SEED, constants.START_SEED + num_runs):

            try:
                run_id = f"Results_{seed}_{int(constants.NUM_MONTHS)}_months_{constants.ARRIVAL_INCREASE_FACTOR}"
                mean_values, min_values, max_values, std_values = get_dists(
                    run_id, plot=False)
                error_bars = {'min': mean_values - min_values,
                              'max': max_values - mean_values}
                columns = [
                    'Time to get Berth', 'Restriction In (T)', 'Channel In', 'Time to get Pilot In',
                    'Time to get Tugs In', 'Terminal Ops', 'Terminal Other', 'Restriction Out (T)',
                    'Time to get Pilot Out', 'Time to get Tugs Out', 'Channel Out'
                ]
                means = mean_values.loc[terminal]
                min_err = error_bars['min'].loc[terminal]
                max_err = error_bars['max'].loc[terminal]
                stds = std_values.loc[terminal]

                means_for_all_seeds.append(means)
                stds_for_all_seeds.append(stds)

                plt.errorbar(columns, means, yerr=[
                             min_err, max_err], fmt='-o', capsize=5, label=f'Seed {seed}', color=colors[seed - constants.START_SEED])

            except Exception as e:
                print(f"Error in processing seed {seed} {e}")
                print("Error:", e)
                pass

        plt.xticks(rotation=90)
        plt.title(f'{terminal} Terminal: Mean with Min/Max and SD')
        plt.xlabel('Processes')
        plt.ylabel('Time (hr)')
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            f'collatedResults/{logfileid}/distPlots/{terminal}_dists.pdf')
        plt.close()

        # plot the overall mean and std for each process
        plt.figure(figsize=(10, 6))
        means_for_all_seeds = np.array(means_for_all_seeds)
        stds_for_all_seeds = np.array(stds_for_all_seeds)
        mean = means_for_all_seeds.mean(axis=0)
        std = stds_for_all_seeds.mean(axis=0)
        # annotate
        for i, (m, s) in enumerate(zip(mean, std)):
            plt.annotate(
                f'Mean: {m:.1f}\nSD: {s:.1f}',
                (i, m),
                textcoords="offset points",
                xytext=(5, 15),
                ha='center'
            )
        plt.errorbar(columns, mean, yerr=std, fmt='-o',
                     capsize=5, label='Mean', color='black')
        plt.xticks(rotation=90)
        plt.title(f'Overall Mean with SD for {terminal} Terminal')
        plt.xlabel('Processes')
        plt.ylabel('Time (hr)')
        plt.tight_layout()
        plt.legend()
        plt.savefig(
            f'collatedResults/{logfileid}/distPlots/overall_dists_{terminal}.pdf')
        plt.close()

    # df to store results for different seeds
    dwell = pd.DataFrame()
    turn = pd.DataFrame()
    anchorage = pd.DataFrame()
    cargo = pd.DataFrame()

    ctr_dwell = pd.DataFrame()
    ctr_turn = pd.DataFrame()
    ctr_anchorage = pd.DataFrame()
    ctr_cargo = pd.DataFrame()

    liq_dwell = pd.DataFrame()
    liq_turn = pd.DataFrame()
    liq_anchorage = pd.DataFrame()
    liq_cargo = pd.DataFrame()

    db_dwell = pd.DataFrame()
    db_turn = pd.DataFrame()
    db_anchorage = pd.DataFrame()
    db_cargo = pd.DataFrame()

    container_queue = pd.DataFrame()
    liquid_queue = pd.DataFrame()
    drybulk_queue = pd.DataFrame()
    all_queue = pd.DataFrame()

    channel = pd.DataFrame()

    ships_entered = []
    ships_exited = []
    ships_entered_channel = []

    ctr_ships_entered = []
    ctr_ships_exited = []
    ctr_ships_entered_channel = []

    liq_ships_entered = []
    liq_ships_exited = []
    liq_ships_entered_channel = []

    db_ships_entered = []
    db_ships_exited = []
    db_ships_entered_channel = []

    total_throughput = []
    ctr_throughput = []
    liq_throughput = []
    dblk_throughput = []

    for seed in tqdm(range(constants.START_SEED, constants.START_SEED + num_runs)):

        try:
            # Wait times
            run_id = f"Results_{seed}_{int(constants.NUM_MONTHS)}_months_{constants.ARRIVAL_INCREASE_FACTOR}"
            df_seed = pd.read_excel(f'.{run_id}/logs/ship_logs.xlsx') 

            ships_entered.append(len(df_seed['Start Time'].dropna()))
            ships_exited.append(len(df_seed['End Time'].dropna()))
            ships_entered_channel.append(
                len(df_seed['Time to Common Channel In'].dropna()))

            # Throughput is computed using rows for each ship the number of container/tons etc loaded + unload and sum for all rows (ships)

            logs_fp = f'.{run_id}/logs/ship_logs.xlsx'
            data_fp = f'.{run_id}/logs/ship_data.csv'

            out = compute_throughput(
            logs_fp,
            data_fp,
            type_map = {"DryBulk":"Dry Bulk"},
            )

            # Append totals
            total_throughput.append(out["total"])
            ctr_throughput.append(out["by_type"].get("Container", 0.0))
            liq_throughput.append(out["by_type"].get("Liquid", 0.0))
            dblk_throughput.append(out["by_type"].get("Dry Bulk", 0.0))

            # print(f"Run {seed}: Total Throughput = {out['total']}")
            # print(f"Run {seed}: Container Throughput = {out['by_type'].get('Container', 0.0)}")
            # print(f"Run {seed}: Liquid Throughput = {out['by_type'].get('Liquid', 0.0)}")
            # print(f"Run {seed}: Dry Bulk Throughput = {out['by_type'].get('Dry Bulk', 0.0)}")

            ctr_ships_entered.append(
                len(df_seed[df_seed['Terminal Type'] == 'Container']['Start Time'].dropna()))
            ctr_ships_exited.append(
                len(df_seed[df_seed['Terminal Type'] == 'Container']['End Time'].dropna()))
            ctr_ships_entered_channel.append(len(
                df_seed[df_seed['Terminal Type'] == 'Container']['Time to Common Channel In'].dropna()))

            liq_ships_entered.append(
                len(df_seed[df_seed['Terminal Type'] == 'Liquid']['Start Time'].dropna()))
            liq_ships_exited.append(
                len(df_seed[df_seed['Terminal Type'] == 'Liquid']['End Time'].dropna()))
            liq_ships_entered_channel.append(len(
                df_seed[df_seed['Terminal Type'] == 'Liquid']['Time to Common Channel In'].dropna()))

            db_ships_entered.append(
                len(df_seed[df_seed['Terminal Type'] == 'Dry Bulk']['Start Time'].dropna()))
            db_ships_exited.append(
                len(df_seed[df_seed['Terminal Type'] == 'Dry Bulk']['End Time'].dropna()))
            db_ships_entered_channel.append(len(
                df_seed[df_seed['Terminal Type'] == 'Dry Bulk']['Time to Common Channel In'].dropna()))

            df_seed = df_seed[WARMUP_ITERS:]

            dwell[f'Dwell_{seed}'] = df_seed['Dwell Time']
            turn[f'Turn_{seed}'] = df_seed['Turn Time']
            anchorage[f'Anchorage_{seed}'] = df_seed['Anchorage Time']
            cargo[f'Cargo_{seed}'] = df_seed['Unloading Time'] + \
                df_seed['Loading Time']

            ctr_dwell[f'Dwell_{seed}'] = df_seed[df_seed['Terminal Type']
                                                 == 'Container']['Dwell Time']
            ctr_turn[f'Turn_{seed}'] = df_seed[df_seed['Terminal Type']
                                               == 'Container']['Turn Time']
            ctr_anchorage[f'Anchorage_{seed}'] = df_seed[df_seed['Terminal Type']
                                                         == 'Container']['Anchorage Time']
            ctr_cargo[f'Cargo_{seed}'] = df_seed[df_seed['Terminal Type'] == 'Container']['Unloading Time'] + \
                df_seed[df_seed['Terminal Type']
                        == 'Container']['Loading Time']

            liq_dwell[f'Dwell_{seed}'] = df_seed[df_seed['Terminal Type']
                                                 == 'Liquid']['Dwell Time']
            liq_turn[f'Turn_{seed}'] = df_seed[df_seed['Terminal Type']
                                               == 'Liquid']['Turn Time']
            liq_anchorage[f'Anchorage_{seed}'] = df_seed[df_seed['Terminal Type']
                                                         == 'Liquid']['Anchorage Time']
            liq_cargo[f'Cargo_{seed}'] = df_seed[df_seed['Terminal Type'] ==
                                                 'Liquid']['Unloading Time'] + df_seed[df_seed['Terminal Type'] == 'Liquid']['Loading Time']

            db_dwell[f'Dwell_{seed}'] = df_seed[df_seed['Terminal Type']
                                                == 'Dry Bulk']['Dwell Time']
            db_turn[f'Turn_{seed}'] = df_seed[df_seed['Terminal Type']
                                              == 'Dry Bulk']['Turn Time']
            db_anchorage[f'Anchorage_{seed}'] = df_seed[df_seed['Terminal Type']
                                                        == 'Dry Bulk']['Anchorage Time']
            db_cargo[f'Cargo_{seed}'] = df_seed[df_seed['Terminal Type'] == 'Dry Bulk']['Unloading Time'] + \
                df_seed[df_seed['Terminal Type'] == 'Dry Bulk']['Loading Time']

            # Anchorage queue
            combined = pd.read_csv(
                f'.{run_id}/logs/waiting_in_anchorage_Container.csv')
            container_queue[f'ctr_{seed}'] = combined['waiting_in_anchorage']

            combined = pd.read_csv(
                f'.{run_id}/logs/waiting_in_anchorage_Liquid.csv')
            liquid_queue[f'liq_{seed}'] = combined['waiting_in_anchorage']

            combined = pd.read_csv(
                f'.{run_id}/logs/waiting_in_anchorage_DryBulk.csv')
            drybulk_queue[f'db_{seed}'] = combined['waiting_in_anchorage']

            combined = pd.read_csv(
                f'.{run_id}/logs/waiting_in_anchorage_all.csv')
            all_queue[f'all_{seed}'] = combined['waiting_in_anchorage']

            # Channel utilization
            total_ships_in_channel = pd.read_csv(
                f'./.{run_id}/logs/total_ships_in_channel.csv')
            channel[f'channel_{seed}'] = total_ships_in_channel.iloc[:, 1]
        except Exception as e:
            print(f"Error in processing seed {seed} {e}")
            with open(f'collatedResults/{logfileid}/run_details_{logfileid}.txt', 'a') as f:
                f.write(f"Error in processing seed {seed} {e})\n")

    print("\n")
    print("Mean ships entered:", sum(ships_entered)/len(ships_entered))
    print("Mean ships exited:", sum(ships_exited)/len(ships_exited))
    print("Mean ships entered channel:", sum(
        ships_entered_channel)/len(ships_entered_channel))
    print("\n")

    print(f"Mean annualised throughput (million tons): {round(((((sum(ctr_throughput)/len(ctr_throughput)) * (sum(constants.CONTAINER_CONVERSION_FACTORS)/len(constants.CONTAINER_CONVERSION_FACTORS))) + ((sum(liq_throughput)/len(liq_throughput)) / (sum(constants.LIQUID_CONVERSION_FACTORS)/len(constants.LIQUID_CONVERSION_FACTORS))) + (sum(dblk_throughput)/len(dblk_throughput))) * (12/constants.NUM_MONTHS)) / 1e6, 2)}")
    print("\n")
    print(f"Mean annualised container throughput (million FEU containers): {round(((sum(ctr_throughput)/len(ctr_throughput)) * (12/constants.NUM_MONTHS)) / 1e6, 2)}")
    print(f"Mean annualised container throughput (million tons): {round(((sum(ctr_throughput)/len(ctr_throughput)) * (12/constants.NUM_MONTHS)) * (sum(constants.CONTAINER_CONVERSION_FACTORS)/len(constants.CONTAINER_CONVERSION_FACTORS)) / 1e6, 2)}")
    print("\n")
    print(f"Mean annualised liquid throughput (million cbm): {round(((sum(liq_throughput)/len(liq_throughput)) * (12/constants.NUM_MONTHS)) / 1e6, 2)}")
    print(f"Mean annualised liquid throughput (million tons): {round(((sum(liq_throughput)/len(liq_throughput)) * (12/constants.NUM_MONTHS)) * (sum(constants.LIQUID_CONVERSION_FACTORS)/len(constants.LIQUID_CONVERSION_FACTORS)) / 1e6, 2)}")
    print("\n")
    print(f"Mean annualised dry bulk throughput (million tons): {round(((sum(dblk_throughput)/len(dblk_throughput)) * (12/constants.NUM_MONTHS)) / 1e6, 2)}")

    with open(f'collatedResults/{logfileid}/run_details_{logfileid}.txt', 'a') as f:
        f.write("\n")
        f.write(
            f"Mean ships entered: {sum(ships_entered)/len(ships_entered)}\n")
        f.write(f"Mean ships exited: {sum(ships_exited)/len(ships_exited)}\n")
        f.write(
            f"Mean ships entered channel: {sum(ships_entered_channel)/len(ships_entered_channel)}\n")
        f.write("\n")
        f.write(
            f"Mean ships entered for container: {sum(ctr_ships_entered)/len(ctr_ships_entered)}\n")
        f.write(
            f"Mean ships exited for container: {sum(ctr_ships_exited)/len(ctr_ships_exited)}\n")
        f.write(
            f"Mean ships entered channel for container: {sum(ctr_ships_entered_channel)/len(ctr_ships_entered_channel)}\n")
        f.write("\n")
        f.write(
            f"Mean ships entered for liquid: {sum(liq_ships_entered)/len(liq_ships_entered)}\n")
        f.write(
            f"Mean ships exited for liquid: {sum(liq_ships_exited)/len(liq_ships_exited)}\n")
        f.write(
            f"Mean ships entered channel for liquid: {sum(liq_ships_entered_channel)/len(liq_ships_entered_channel)}\n")
        f.write("\n")
        f.write(
            f"Mean ships entered for dry bulk: {sum(db_ships_entered)/len(db_ships_entered)}\n")
        f.write(
            f"Mean ships exited for dry bulk: {sum(db_ships_exited)/len(db_ships_exited)}\n")
        f.write(
            f"Mean ships entered channel for dry bulk: {sum(db_ships_entered_channel)/len(db_ships_entered_channel)}\n")
        f.write("\n")

        f.write(f"Mean annualised throughput (million tons): {round(((((sum(ctr_throughput)/len(ctr_throughput)) * (sum(constants.CONTAINER_CONVERSION_FACTORS)/len(constants.CONTAINER_CONVERSION_FACTORS))) + ((sum(liq_throughput)/len(liq_throughput)) / (sum(constants.LIQUID_CONVERSION_FACTORS)/len(constants.LIQUID_CONVERSION_FACTORS))) + (sum(dblk_throughput)/len(dblk_throughput))) * (12/constants.NUM_MONTHS)) / 1e6, 2)}\n")
        f.write(f"Mean annualised container throughput (million FEU containers): {round(((sum(ctr_throughput)/len(ctr_throughput)) * (12/constants.NUM_MONTHS)) / 1e6, 2)}\n")
        f.write(f"Mean annualised container throughput (million tons): {round(((sum(ctr_throughput)/len(ctr_throughput)) * (12/constants.NUM_MONTHS)) * (sum(constants.CONTAINER_CONVERSION_FACTORS)/len(constants.CONTAINER_CONVERSION_FACTORS)) / 1e6, 2)}\n")
        f.write(f"Mean annualised liquid throughput (million cbm): {round(((sum(liq_throughput)/len(liq_throughput)) * (12/constants.NUM_MONTHS)) / 1e6, 2)}\n")
        f.write(f"Mean annualised liquid throughput (million tons): {round(((sum(liq_throughput)/len(liq_throughput)) * (12/constants.NUM_MONTHS)) * (sum(constants.LIQUID_CONVERSION_FACTORS)/len(constants.LIQUID_CONVERSION_FACTORS)) / 1e6, 2)}\n")
        f.write(f"Mean annualised dry bulk throughput (million tons): {round(((sum(dblk_throughput)/len(dblk_throughput)) * (12/constants.NUM_MONTHS)) / 1e6, 2)}\n")


        f.write("\n")

    # Plot multirun combined anchorage queues
    print("\nAnchorage queue lengths:\n")
    with open(f'collatedResults/{logfileid}/run_details_{logfileid}.txt', 'a') as f:
        f.write("\nAnchorage queue lengths:\n")
    list_of_df = [container_queue, liquid_queue, drybulk_queue, all_queue]
    df_name = ['container', 'liquid', 'drybulk', 'all']
    for i, df in enumerate(list_of_df):
        # remove last 5 rows
        df = df[:-5].copy()
        df_analysis = pd.DataFrame()
        ship_type = df_name[i]
        df_analysis['mean'] = df.mean(axis=1)
        df_analysis['std'] = df.std(axis=1)
        df_analysis['max'] = df.max(axis=1)
        df_analysis['min'] = df.min(axis=1)
        df_analysis['rolling_mean'] = df_analysis['mean'].rolling(
            window=200).mean()
        mean = df_analysis['mean'][WARMUP_ITERS:].mean()
        plt.plot(df_analysis['mean'], label='Mean')
        plt.plot(df_analysis['rolling_mean'], label='Rolling Mean')
        plt.fill_between(
            df_analysis.index, df_analysis['min'], df_analysis['max'], color='skyblue', alpha=0.4, label='Min-Max Range')
        plt.fill_between(df_analysis.index, df_analysis['mean'] - df_analysis['std'], df_analysis['mean'] +
                         df_analysis['std'], color='lightgreen', alpha=0.4, label='Mean ± Std Dev')
        plt.axvspan(0, WARMUP_ITERS, color='lightgray',
                    alpha=0.5, label='Warmup Period', zorder=-10)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(left=0)
        plt.title(f'Anchorage queue of {ship_type} ships', fontsize=18)
        plt.xlabel('Time (hr)', fontsize=15)
        plt.ylabel('Number of ships', fontsize=15)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f'collatedResults/{logfileid}/queuePlots/Queue_{ship_type}.pdf')
        # plt.show()
        plt.close()
        print(
            f"Mean (+- std) for anchorage queue of {ship_type} ships: {df_analysis['mean'][WARMUP_ITERS:].mean():.1f} (+- {df_analysis['std'][WARMUP_ITERS:].mean():.1f})")
        with open(f'collatedResults/{logfileid}/run_details_{logfileid}.txt', 'a') as f:
            f.write(
                f"Mean (+- std) for anchorage queue of {ship_type} ships: {df_analysis['mean'][WARMUP_ITERS:].mean():.1f} (+- {df_analysis['std'][WARMUP_ITERS:].mean():.1f})\n")
    print("\n")

    mean_queue_length = df_analysis['mean'][WARMUP_ITERS:].mean()

    # Plot wait and processing times with error bars for multirun output
    print("\nTime taken for different ship types:")
    with open(f'collatedResults/{logfileid}/run_details_{logfileid}.txt', 'a') as f:
        f.write("\nTime taken for different ship types:\n")
    list_of_df = [dwell, turn, anchorage, cargo, ctr_dwell, ctr_turn, ctr_anchorage, ctr_cargo,
                  liq_dwell, liq_turn, liq_anchorage, liq_cargo, db_dwell, db_turn, db_anchorage, db_cargo]
    what = ['dwell', 'turn', 'anchorage', 'cargo'] * 4
    which = ['all'] * 4 + ['container'] * 4 + ['liquid'] * 4 + ['drybulk'] * 4
    for i, df in enumerate(list_of_df):
        ship_type = which[i]
        time_type = what[i]
        df['mean'] = df.mean(axis=1)
        df['std'] = df.std(axis=1)

        # Calculate histogram
        num_bins = 10
        bins = np.linspace(df['mean'].min(), df['mean'].max(), num_bins + 1)
        counts, bin_edges = np.histogram(df['mean'], bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        std_in_bins = []
        for i in range(len(bins) - 1):
            std_in_bins.append(df[(df['mean'] > bins[i]) & (
                df['mean'] <= bins[i + 1])]['std'].mean())

        plt.bar(bin_centers, counts, width=(
            bin_edges[1] - bin_edges[0]) * 0.8, align='center', alpha=0.7, label='Histogram')
        plt.errorbar(bin_centers, counts, yerr=std_in_bins,
                     fmt='o', color='red', label='Error bars')
        plt.xlabel('Time (hr)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {time_type} time for {ship_type} ships')
        plt.legend()
        plt.savefig(
            f'collatedResults/{logfileid}/timePlots/{time_type}_{ship_type}.pdf')
        plt.close()

        print(
            f"Mean (+- std) for {time_type} time for {ship_type} ships: {df['mean'].mean():.1f} (+- {df['std'].mean():.1f})")
        with open(f'collatedResults/{logfileid}/run_details_{logfileid}.txt', 'a') as f:
            f.write(
                f"Mean (+- std) for {time_type} time for {ship_type} ships: {df['mean'].mean():.1f} (+- {df['std'].mean():.1f})\n")

    # Capacity calculation

    # anchorage queue wait
    mean_all_anc_wait = anchorage.mean(axis=1).mean()

    # mean queue lengths
    mean_ctr_queue = container_queue.mean(axis=1).mean()
    mean_liq_queue = liquid_queue.mean(axis=1).mean()
    mean_db_queue = drybulk_queue.mean(axis=1).mean()
    queue_lengths = [mean_ctr_queue, mean_liq_queue, mean_db_queue]

    # arrival rates
    mean_ctr_arrival = constants.mean_interarrival_time_container
    mean_liq_arrival = constants.mean_interarrival_time_tanker
    mean_db_arrival = constants.mean_interarrival_time_gencargo
    arrival_rates = [1/mean_ctr_arrival, 1/mean_liq_arrival, 1/mean_db_arrival]
    current_mean_arrival_rate = sum(arrival_rates)


    print("\nChannel utilization:")
    with open(f'collatedResults/{logfileid}/run_details_{logfileid}.txt', 'a') as f:
        f.write("\nChannel utilization:\n")
    channel_analysis = pd.DataFrame()
    channel_analysis['mean'] = channel.mean(axis=1)
    channel_analysis['std'] = channel.std(axis=1)
    channel_analysis['max'] = channel.max(axis=1)
    channel_analysis['min'] = channel.min(axis=1)
    channel_analysis['rolling_mean'] = channel_analysis['mean'].rolling(
        window=200).mean()
    mean = channel_analysis['mean'][WARMUP_ITERS:].mean()
    plt.plot(channel_analysis['mean'], label='Mean')
    plt.plot(channel_analysis['rolling_mean'], label='Rolling Mean')
    plt.fill_between(channel_analysis.index,
                     channel_analysis['min'], channel_analysis['max'], color='skyblue', alpha=0.4, label='Min-Max Range')
    plt.fill_between(channel_analysis.index, channel_analysis['mean'] - channel_analysis['std'],
                     channel_analysis['mean'] + channel_analysis['std'], color='lightgreen', alpha=0.4, label='Mean ± Std Dev')
    plt.axvspan(0, WARMUP_ITERS, color='lightgray',
                alpha=0.5, label='Warmup Period', zorder=-10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=0)
    plt.title('Channel utilization', fontsize=18)
    plt.xlabel('Time (hr)', fontsize=15)
    plt.ylabel('Number of ships', fontsize=15)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'collatedResults/{logfileid}/channel.pdf')
    plt.close()
    print(
        f"Mean (+- std) for channel utilization: {channel_analysis['mean'][WARMUP_ITERS:].mean():.1f} (+- {channel_analysis['std'][WARMUP_ITERS:].mean():.1f})\n")
    with open(f'collatedResults/{logfileid}/run_details_{logfileid}.txt', 'a') as f:
        f.write(
            f"Mean (+- std) for channel utilization: {channel_analysis['mean'][WARMUP_ITERS:].mean():.1f} (+- {channel_analysis['std'][WARMUP_ITERS:].mean():.1f})\n")

    try:
        output = calculate_mean_wait_time(
            arrival_rates, queue_lengths, mean_all_anc_wait, logfilename)
        op_cap = output['Service Rate']
    except Exception as e:
        print("Operating capacity calculation failed")
        print("Error:", e)
        op_cap = 0.0

    print(f"Operating capacity: {op_cap:.2f} vessels / hr")

    print("SAVING RESULTS")
    if config.SCENARIO_NAME == 'BottleneckDetection':
        print("Bottleneck detection scenario, not saving results")
        if np.isclose(current_mean_arrival_rate, base_arrival_rate, atol=1e-1):
            with open(f'bottleneckAnalysis/logs/{SCENARIO_NAME}_BASE.txt', 'w') as f: #use the scenario name from constants
                f.write(f"Operating capacity: {op_cap:.2f} vessels / hr\n")
                f.write(f"Arrival rate: {current_mean_arrival_rate:.2f} vessels / hr\n")
                f.write(f"Exited ships: {sum(ships_exited)/len(ships_exited):.2f} vessels / hr\n")
                f.write(f"Dwell time (Container): {ctr_dwell['mean'].mean():.1f} hr\n")
                f.write(f"Dwell time (Liquid): {liq_dwell['mean'].mean():.1f} hr\n")
                f.write(f"Dwell time (Dry Bulk): {db_dwell['mean'].mean():.1f} hr\n")
                f.write(f"Turn time (Average): {turn['mean'].mean():.1f} hr\n")
                f.write(f"Anchorage queue length: {mean_queue_length:.2f} ships\n")
                f.write(f"Anchorage queue length (Container): {mean_ctr_queue:.2f} ships\n")
                f.write(f"Anchorage queue length (Liquid): {mean_liq_queue:.2f} ships\n")
                f.write(f"Anchorage queue length (Dry Bulk): {mean_db_queue:.2f} ships\n")      
                f.write(f"Mean wait time in anchorage: {mean_all_anc_wait:.2f} hr\n")
                f.write(f"Mean wait time in anchorage (Container): {mean_ctr_queue:.2f} hr\n")
                f.write(f"Mean wait time in anchorage (Liquid): {mean_liq_queue:.2f} hr\n")
                f.write(f"Mean wait time in anchorage (Dry Bulk): {mean_db_queue:.2f} hr\n")
                f.write(f"Mean channel utilization: {channel_analysis['mean'][WARMUP_ITERS:].mean():.1f} ships\n")
        else:
            with open(f'bottleneckAnalysis/logs/{SCENARIO_NAME}_{ARRIVAL_INCREASE_FACTOR}.txt', 'w') as f: #use the scenario name from constants
                f.write("Arrival rate: {:.2f} vessels / hr\n".format(current_mean_arrival_rate))
                f.write("Exited ships: {:.2f} vessels / hr\n".format(sum(ships_exited)/len(ships_exited)))

    # Cu_opt = calculate_ultimate_capacity()


    try:
        Cu_opt,theta_opt = calculate_ultimate_capacity()
        with open(f'collatedResults/{logfileid}/run_details_{logfileid}.txt', 'a') as f:
            f.write(f"Ultimate capacity: {Cu_opt:.2f} vessels / hr\n")
            f.write(f"Optimal theta: {theta_opt:.2f}\n")
    except Exception as e:
        print("Ultimate capacity calculation failed")
        print("Error:", e)
        print("More simulation runs may be needed to calculate ultimate capacity.")
    

    # save the dataframes
    dwell.to_csv(f'collatedResults/{logfileid}/data/dwell.csv')
    turn.to_csv(f'collatedResults/{logfileid}/data/turn.csv')
    anchorage.to_csv(f'collatedResults/{logfileid}/data/anchorage.csv')
    cargo.to_csv(f'collatedResults/{logfileid}/data/cargo.csv')

    ctr_dwell.to_csv(f'collatedResults/{logfileid}/data/ctr_dwell.csv')
    ctr_turn.to_csv(f'collatedResults/{logfileid}/data/ctr_turn.csv')
    ctr_anchorage.to_csv(f'collatedResults/{logfileid}/data/ctr_anchorage.csv')
    ctr_cargo.to_csv(f'collatedResults/{logfileid}/data/ctr_cargo.csv')

    liq_dwell.to_csv(f'collatedResults/{logfileid}/data/liq_dwell.csv')
    liq_turn.to_csv(f'collatedResults/{logfileid}/data/liq_turn.csv')
    liq_anchorage.to_csv(f'collatedResults/{logfileid}/data/liq_anchorage.csv')
    liq_cargo.to_csv(f'collatedResults/{logfileid}/data/liq_cargo.csv')

    db_dwell.to_csv(f'collatedResults/{logfileid}/data/db_dwell.csv')
    db_turn.to_csv(f'collatedResults/{logfileid}/data/db_turn.csv')
    db_anchorage.to_csv(f'collatedResults/{logfileid}/data/db_anchorage.csv')
    db_cargo.to_csv(f'collatedResults/{logfileid}/data/db_cargo.csv')

    container_queue.to_csv(
        f'collatedResults/{logfileid}/data/container_queue.csv')
    liquid_queue.to_csv(f'collatedResults/{logfileid}/data/liquid_queue.csv')
    drybulk_queue.to_csv(f'collatedResults/{logfileid}/data/drybulk_queue.csv')
    all_queue.to_csv(f'collatedResults/{logfileid}/data/all_queue.csv')

    channel.to_csv(f'collatedResults/{logfileid}/data/channel.csv')

    # restrictions

    # Replace these with your actual list of run_ids
    run_ids = []
    for seed in range(constants.START_SEED, constants.START_SEED + num_runs):
        try:
            run_id = f"Results_{seed}_{int(constants.NUM_MONTHS)}_months_{constants.ARRIVAL_INCREASE_FACTOR}"
            run_ids.append(run_id)
        except:
            print(f"Error in processing seed {seed} (does not exist)")
            continue

    in_path_template = '.{}/bottlenecks/Waterway/in_hist_percentage.csv'
    out_path_template = '.{}/bottlenecks/Waterway/out_hist_percentage.csv'
    in_hist_list = [pd.read_csv(in_path_template.format(
        run_id), index_col=0) for run_id in run_ids]
    out_hist_list = [pd.read_csv(out_path_template.format(
        run_id), index_col=0) for run_id in run_ids]

    restriction_types = ['Beam', 'Draft', 'Daylight', 'Total']

    def compute_mean_std(hist_list, suffix):
        means = {}
        stds = {}
        for r in restriction_types:
            col = f"{r} {suffix}"
            # shape: (bins, seeds)
            stacked = np.stack([df[col].values for df in hist_list], axis=1)
            means[r] = np.mean(stacked, axis=1)
            stds[r] = np.std(stacked, axis=1)
        bin_labels = hist_list[0].index
        return means, stds, bin_labels

    mean_in, std_in, index_in = compute_mean_std(in_hist_list, 'In')
    mean_out, std_out, index_out = compute_mean_std(out_hist_list, 'Out')

    def save_individual_errorbar_pdfs(mean_dict, std_dict, index_labels, suffix, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        x = np.arange(len(index_labels))
        xticks = ['0' if str(label) == '(-1, 0]' else str(label)
                  for label in index_labels]

        for r in restriction_types:
            mean_vals = mean_dict[r]
            std_vals = std_dict[r]

            plt.figure(figsize=(10, 6))
            plt.bar(x, mean_vals, yerr=std_vals, capsize=5, alpha=0.7)
            plt.xticks(ticks=x, labels=xticks, rotation=45)
            plt.ylabel('Percentage', fontsize=12)
            plt.xlabel('Waiting Time (hr)', fontsize=12)
            plt.title(
                f'{r} waiting time distribution ({suffix.lower()})', fontsize=14)
            plt.tight_layout()

            pdf_path = os.path.join(
                output_folder, f'{r.lower()}_{suffix.lower()}_errorbar.pdf')
            plt.savefig(pdf_path)
            plt.close()

    output_folder = f'collatedResults/{logfileid}/restrictionPlots/'

    save_individual_errorbar_pdfs(
        mean_in, std_in, index_in, 'In', output_folder)
    save_individual_errorbar_pdfs(
        mean_out, std_out, index_out, 'Out', output_folder)

    collate_results_time = round((time.time() - start_time) / 60, 2)

    with open(f'collatedResults/{logfileid}/run_details_{logfileid}.txt', 'a') as f:
        f.write(
            f"\nTotal simulation runtime: {round(total_time + collate_results_time, 2)} minutes\n")

    print(
        f"Total time taken: {round(total_time + collate_results_time, 2)} minutes\n")

def compute_throughput(
    logs_fp: str,
    data_fp: str,
    *,
    strict_missing: bool = True,
    type_map: dict | None = None,
) -> dict:
    """
    Compute throughput for a single run given paths to:
      - logs_fp: Excel/CSV with 'Ship_Id' and 'End Time'
      - data_fp: CSV with 'ship_id', 'ship_type',
                 'num_container_or_liq_tons_or_dry_tons_to_load',
                 'num_container_or_liq_tons_or_dry_tons_to_unload'

    Returns:
      {
        "total": float,
        "by_type": { "<type>": float, ... },
        "ships_exited": int,
        "exited_ids": list[str]
      }

    Raises:
      KeyError / ValueError with clear messages if required columns are missing
      or (if strict_missing=True) any exited ship isn't found in ship_data.
    """
    # --- read ---
    if logs_fp.lower().endswith((".xls", ".xlsx")):
        df_logs = pd.read_excel(logs_fp)
    else:
        df_logs = pd.read_csv(logs_fp)

    df_data = pd.read_csv(data_fp)

    # --- sanity: required columns ---
    req_logs = {"Ship_Id", "End Time"}
    req_data = {
        "ship_id",
        "ship_type",
        "num_container_or_liq_tons_or_dry_tons_to_load",
        "num_container_or_liq_tons_or_dry_tons_to_unload",
    }
    missing_logs = req_logs - set(df_logs.columns)
    missing_data = req_data - set(df_data.columns)
    if missing_logs:
        raise KeyError(f"Ship Logs missing columns: {sorted(missing_logs)}")
    if missing_data:
        raise KeyError(f"Ship data missing columns: {sorted(missing_data)}")

    # --- normalize ids ---
    # logs are 0-based ("Ship_0", "Ship_1", ...)
    df_logs["Ship_Id"] = (
        df_logs["Ship_Id"]
        .str.extract(r"(\d+)")[0]   # pull out the number part
        .astype(int)
        .add(1)                     # shift to match ship_data (1-based)
        .astype(str)
    )

    df_logs["Ship_Id"] = df_logs["Ship_Id"].astype(int)
    df_data["ship_id"] = df_data["ship_id"].astype(int)

    # --- exited ships set ---
    exited_ids = (
        df_logs.loc[df_logs["End Time"].notna(), "Ship_Id"]
        .dropna()
        .unique()
        .tolist()
    )
    ships_exited = len(exited_ids)

    # --- subset data to exited; validate coverage ---
    data_exited = df_data[df_data["ship_id"].isin(exited_ids)].copy()
    missing = sorted(set(exited_ids) - set(data_exited["ship_id"]))
    if missing and strict_missing:
        raise ValueError(
            f"{len(missing)} exited ship(s) not found in ship_data: "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )

    # --- (optional) standardize type labels ---
    if type_map:
        data_exited["ship_type"] = data_exited["ship_type"].map(
            lambda x: type_map.get(x, x)
        )

    # --- numeric quantities ---
    for c in [
        "num_container_or_liq_tons_or_dry_tons_to_load",
        "num_container_or_liq_tons_or_dry_tons_to_unload",
    ]:
        data_exited[c] = pd.to_numeric(data_exited[c], errors="coerce").fillna(0)

    # --- collapse to one row per ship_id (+ keep ship_type) ---
    grouped = (
        data_exited.groupby(["ship_id", "ship_type"], as_index=False)[
            [
                "num_container_or_liq_tons_or_dry_tons_to_load",
                "num_container_or_liq_tons_or_dry_tons_to_unload",
            ]
        ]
        .sum()
    )
    grouped["ship_throughput"] = (
        grouped["num_container_or_liq_tons_or_dry_tons_to_load"]
        + grouped["num_container_or_liq_tons_or_dry_tons_to_unload"]
    )

    # --- aggregates ---
    by_type = grouped.groupby("ship_type")["ship_throughput"].sum().to_dict()
    total = float(grouped["ship_throughput"].sum())

    return {
        "total": float(total),
        "by_type": {k: float(v) for k, v in by_type.items()},
        "ships_exited": ships_exited,
        "exited_ids": exited_ids,
    }

# collate_results(5, 8)
