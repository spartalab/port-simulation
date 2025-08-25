"""
This script runs a series of simulations to analyze the breakpoint of a maritime system.
It updates the constants file by adjusting the arrival rate multipliers, runs the simulation, extracts data from the results,
and fits an ODE model to the exit data. The results are then plotted and saved for further simulation_analysis.
This is a standalone script that can be run independently.
It is designed to be used in conjunction with a simulation framework that uses the constants defined in `constants.py`.
"""
import subprocess
import os
import re
import sys
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from tqdm import tqdm
import importlib

shutil.copy('inputs/terminal_data_base.csv',  'inputs/terminal_data.csv')
shutil.copy('config.py', 'constants.py')

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(parent_dir)
import constants
CONSTANTS_FILE_PATH ="./constants.py"


from simulation_analysis.capacity import solve_N


def objective(params, lambda_data, exits_data):
    """
    Objective function for ODE model fitting.
    Args:
        params (tuple): Model parameters (Cu, theta).
        lambda_data (np.ndarray): Array of arrival rates (λ).
        exits_data (np.ndarray): Array of observed exits data.

    Returns:
        float: Mean squared error between model prediction and observed exits.
    """
    T = constants.SIMULATION_TIME
    Cu, theta = params
    if Cu <= 0 or theta <= 0:
        return np.inf
    N_model = solve_N(lambda_data, Cu, T, theta)
    return np.mean((exits_data - N_model)**2)


def fit_ODE_model(df):
    """
    Fit the ODE model to the exit data from the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing 'lambda' and 'exits' columns.
    Returns:
        tuple: Fitted parameters (Cu_fit, theta_fit).
    """
    T = constants.SIMULATION_TIME
    lambda_data = df['lambda'].values
    exits_data = df['exits'].values
    initial_guess = [np.max(exits_data), 2.0]

    res = minimize(objective, initial_guess, args=(
        lambda_data, exits_data), method='Nelder-Mead')
    Cu_fit, theta_fit = res.x                   

    N_fit = solve_N(lambda_data, Cu_fit, T, theta_fit)

    # Compute metrics
    obj_value = objective(res.x, lambda_data, exits_data)
    ss_res = np.sum((exits_data - N_fit)**2)
    ss_tot = np.sum((exits_data - np.mean(exits_data))**2)
    r2 = 1 - ss_res / ss_tot  

    # Print results
    print(f"RMSE: {np.sqrt(obj_value):.4f}") 
    print(f"R²: {r2:.4f}")  
    print(f"Cu_fit: {Cu_fit:.4f}")
    print(f"theta_fit: {theta_fit:.4f}")

    plt.figure(figsize=(5, 4))
    plt.scatter(lambda_data, exits_data,
                label='Observed', color='blue', s=10)
    plt.plot(lambda_data, N_fit,
            label='ODE Fit', linewidth=3, color='orange')
    plt.xlabel('λ', fontsize=14)
    plt.ylabel('$N^{exits}$', fontsize=14)
    plt.axhline(y=T * Cu_fit, color='m', linestyle='--', label='Ultimate capacity')
    plt.title('ODE Model Fit', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=0, right=np.max(lambda_data) * 1.1)
    plt.ylim(bottom=0, top=np.max(exits_data) * 1.1)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  
    plt.figtext(
        0.55,     # x
        0.05,    # y
        f'$C_u$: {Cu_fit:.2f}, θ: {theta_fit:.2f}',
        ha='center',
        fontsize=14,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3')
    )
    plt.savefig(f"./breakpointAnalysis/{file_id}_ODE_fit.pdf")
    plt.show()
    plt.close()

    return Cu_fit, theta_fit


def update_constants_file(log_num, inter_arrival_factors):
    """
    Update the constants.py file with new values for LOG_NUM and inter-arrival factors.
    Args:
        log_num (int): The log number to be set in the constants file.
        inter_arrival_factors (tuple): A tuple containing the inter-arrival factors for container, liquid, and drybulk ships.
    """
    arrival_factor_ctr, arrival_factor_liq, arrival_factor_drybulk = inter_arrival_factors

    with open(CONSTANTS_FILE_PATH, 'r') as f:
        lines = f.readlines()

    with open(CONSTANTS_FILE_PATH, 'w') as f:
        for line in lines:
            if line.startswith('LOG_NUM'):
                f.write(f'LOG_NUM = {log_num}\n')
            elif line.startswith('ARRIVAL_INCREASE_FACTOR_CTR'):
                f.write(
                    f'ARRIVAL_INCREASE_FACTOR_CTR = {arrival_factor_ctr}\n')
            elif line.startswith('ARRIVAL_INCREASE_FACTOR_LIQ'):
                f.write(
                    f'ARRIVAL_INCREASE_FACTOR_LIQ = {arrival_factor_liq}\n')
            elif line.startswith('ARRIVAL_INCREASE_FACTOR_DRYBULK'):
                f.write(
                    f'ARRIVAL_INCREASE_FACTOR_DRYBULK = {arrival_factor_drybulk}\n')
            else:
                f.write(line)
    
    # reload the constants module to reflect changes
    importlib.reload(constants)


def run_simulation(params):
    """
    Run the simulation with the given parameters.
    Args:
        params (tuple): A tuple containing the log number and inter-arrival factors.
    """
    log_num, inter_arrival_factors = params
    update_constants_file(log_num, inter_arrival_factors)

    try:
        # Run the simulation with updated constants.py
        subprocess.run([sys.executable, 'main.py'], check=True)
    except Exception as e:
        print(f"Error running simulation for LOG_NUM={log_num}: {e}")


def extract_data(log_ids, simulation_time, base_path):
    """
    Extract relevant data from the simulation log files.
    Args:
        log_ids (list): List of log IDs to process.
        simulation_time (int): Total simulation time in seconds.
        base_path (str): Base path where the log files are stored.
    Returns:
        dict: A dictionary containing extracted data for container, liquid, drybulk ships, channel utilization, and mean entered/exited ships.
    """
    data = {
        "container": [],
        "liquid": [],
        "drybulk": [],
        "channel_utilization": [],
        "mean_entered": [],
        "mean_exited": [],
        "mean_ctr_entered": [],
        "mean_ctr_exited": [],
        "mean_liq_entered": [],
        "mean_liq_exited": [],
        "mean_drybulk_entered": [],
        "mean_drybulk_exited": []
    }

    months = int(simulation_time / (30 * 24))

    for log_id in log_ids:
        # Construct the file path
        id = f'{months}_months_{constants.NUM_RUNS}_runs_run_{log_id}'
        file_path = f"{base_path}/{id}/run_details_{id}.txt"

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Read the file and extract relevant data
        with open(file_path, 'r') as file:

            content = file.read()

            # Regex patterns for anchorage queue lengths
            patterns = {
                "container": r"Mean \(\+- std\) for anchorage queue of container ships:\s*([\d.]*)\s*\(\+-\s*([\d.]*)\)?",
                "liquid": r"Mean \(\+- std\) for anchorage queue of liquid ships:\s*([\d.]*)\s*\(\+-\s*([\d.]*)\)?",
                "drybulk": r"Mean \(\+- std\) for anchorage queue of drybulk ships:\s*([\d.]*)\s*\(\+-\s*([\d.]*)\)?"
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    try:
                        mean, std = map(float, match.groups())
                        data[key].append((mean, std))
                    except:
                        mean = float(match.group(1))
                        std = 0
                        data[key].append((mean, std))

            # Regex for channel utilization mean
            channel_util_pattern = r"Mean \(\+- std\) for channel utilization:\s*([\d.]*)\s*\(\+-\s*([\d.]*)\)?"
            channel_util_match = re.search(channel_util_pattern, content)
            if channel_util_match:
                try:
                    mean, std = map(float, channel_util_match.groups())
                except:
                    mean = float(channel_util_match.group(1))
                    std = 0
                data["channel_utilization"].append((mean, std))

            # Regex for mean entered and exited ships
            mean_entered_pattern = r"Mean ships entered: ([\d.]+)"
            mean_exited_pattern = r"Mean ships entered channel: ([\d.]+)"
            mean_ctr_entered_pattern = r"Mean ships entered for container: ([\d.]+)"
            mean_ctr_exited_pattern = r"Mean ships entered channel for container: ([\d.]+)"
            mean_liq_entered_pattern = r"Mean ships entered for liquid: ([\d.]+)"
            mean_liq_exited_pattern = r"Mean ships entered channel for liquid: ([\d.]+)"
            mean_drybulk_entered_pattern = r"Mean ships entered for dry bulk: ([\d.]+)"
            mean_drybulk_exited_pattern = r"Mean ships entered channel for dry bulk: ([\d.]+)"

            entered_match = re.search(mean_entered_pattern, content)
            exited_match = re.search(mean_exited_pattern, content)
            ctr_entered_match = re.search(mean_ctr_entered_pattern, content)
            ctr_exited_match = re.search(mean_ctr_exited_pattern, content)
            liq_entered_match = re.search(mean_liq_entered_pattern, content)
            liq_exited_match = re.search(mean_liq_exited_pattern, content)
            drybulk_entered_match = re.search(
                mean_drybulk_entered_pattern, content)
            drybulk_exited_match = re.search(
                mean_drybulk_exited_pattern, content)

            if entered_match:
                data["mean_entered"].append(float(entered_match.group(1)))
            if exited_match:
                data["mean_exited"].append(float(exited_match.group(1)))
            if ctr_entered_match:
                data["mean_ctr_entered"].append(
                    float(ctr_entered_match.group(1)))
            if ctr_exited_match:
                data["mean_ctr_exited"].append(
                    float(ctr_exited_match.group(1)))
            if liq_entered_match:
                data["mean_liq_entered"].append(
                    float(liq_entered_match.group(1)))
            if liq_exited_match:
                data["mean_liq_exited"].append(
                    float(liq_exited_match.group(1)))
            if drybulk_entered_match:
                data["mean_drybulk_entered"].append(
                    float(drybulk_entered_match.group(1)))
            if drybulk_exited_match:
                data["mean_drybulk_exited"].append(
                    float(drybulk_exited_match.group(1)))

    return data


def plot_data(data, plot_error_bars=True):
    """
    Plot the extracted data and save the plots to files.
    The function generates several plots including:
    - Mean entered vs mean exited ships
    - Mean entered - exited ships
    - Mean entered vs mean exited container ships
    - Mean entered vs mean exited liquid ships
    - Mean entered vs mean exited drybulk ships
    - Channel utilization with error bars
    - Anchorage queue lengths for container, liquid, and drybulk ships
    - Anchorage queue lengths for all ships together
    - Anchorage queue lengths for all ships together (stable range)
    Args:
        data (dict): Dictionary containing extracted data.
        plot_error_bars (bool): Whether to plot error bars for the channel utilization.
    """

    arrival_factors = [x for x in inter_arrival_factors_list]

    # Plot 1: Entered vs Exited Ships
    plt.plot(arrival_factors, data['mean_entered'],
             label="Mean Entered Ships", marker='o')
    plt.plot(arrival_factors, data['mean_exited'],
             label="Mean Exited Ships", marker='s')
    plateau_avg = sum(data['mean_exited'][-4:]) / 4
    plt.axhline(y=plateau_avg, color='r', linestyle='--',
                label=f"At capacity: {plateau_avg:.2f}")
    plt.title("Mean entered vs mean exited ships")
    plt.xlabel("Arrival rate multiplier")
    plt.ylabel("Number of ships")
    plt.legend()
    plt.grid()
    plt.savefig(f"./breakpointAnalysis/{file_id}_throughput.pdf")
    plt.close()

    output_file = f"./breakpointAnalysis/{file_id}_throughput.csv"
    with open(output_file, 'w') as f:
        f.write("Arrival Rate,Mean Entered,Mean Exited\n")
        for i in range(len(arrival_factors)):
            f.write(
                f"{arrival_factors[i]},{data['mean_entered'][i]},{data['mean_exited'][i]}\n")
    print(f"Data saved to {output_file}")

    k = 40
    plt.plot(arrival_factors[:k], data['mean_entered']
             [:k], label="Mean Entered Ships", marker='o')
    plt.plot(arrival_factors[:k], data['mean_exited']
             [:k], label="Mean Exited Ships", marker='s')
    plt.title("Mean entered vs mean exited ships for stable range")
    plt.xlabel("Arrival rate multiplier")
    plt.ylabel("Number of ships")
    plt.legend()
    plt.grid()
    plt.savefig(f"./breakpointAnalysis/{file_id}_throughput_stable.pdf")
    plt.close()

    # Plot 2: entered - exited ships
    plt.plot(arrival_factors, np.array(data['mean_entered']) - np.array(
        data['mean_exited']), label="Mean Entered - Exited Ships", marker='o')
    plateau_avg = sum(
        np.array(data['mean_entered'][:3]) - np.array(data['mean_exited'][:3])) / 3

    plt.axhline(y=plateau_avg, color='r', linestyle='--',
                label=f"Stable system: {plateau_avg:.2f}")
    plt.title("Mean entered - exited ships")
    plt.xlabel("Arrival rate multiplier")
    plt.ylabel("Number of ships")
    plt.legend()
    plt.grid()
    plt.savefig(
        f"./breakpointAnalysis/{file_id}_throughput_entered_exited.pdf")
    plt.close()

    # Plot 3: ctr ships entered vs exited
    plt.plot(arrival_factors, data['mean_ctr_entered'],
             label="Mean Entered Container Ships", marker='o')
    plt.plot(arrival_factors, data['mean_ctr_exited'],
             label="Mean Exited Container Ships", marker='s')
    plateau_avg = sum(data['mean_ctr_exited'][-3:]) / 3
    plt.axhline(y=plateau_avg, color='r', linestyle='--',
                label=f"At capacity: {plateau_avg:.2f}")
    plt.title("Mean entered vs mean exited container ships")
    plt.xlabel("Arrival rate multiplier")
    plt.ylabel("Number of container ships")
    plt.legend()
    plt.grid()
    plt.savefig(f"./breakpointAnalysis/{file_id}_throughput_ctr.pdf")
    plt.close()

    # Plot 4: liq ships entered vs exited
    plt.plot(arrival_factors, data['mean_liq_entered'],
             label="Mean Entered Liquid Ships", marker='o')
    plt.plot(arrival_factors, data['mean_liq_exited'],
             label="Mean Exited Liquid Ships", marker='s')
    plateau_avg = sum(data['mean_liq_exited'][-4:]) / 4
    plt.axhline(y=plateau_avg, color='r', linestyle='--',
                label=f"At capacity: {plateau_avg:.2f}")
    plt.title("Mean entered vs mean exited liquid ships")
    plt.xlabel("Arrival rate multiplier")
    plt.ylabel("Number of liquid ships")
    plt.legend()
    plt.grid()
    plt.savefig(f"./breakpointAnalysis/{file_id}_throughput_liq.pdf")
    plt.close()

    # Plot 5: drybulk ships entered vs exited
    plt.plot(arrival_factors, data['mean_drybulk_entered'],
             label="Mean Entered Drybulk Ships", marker='o')
    plt.plot(arrival_factors, data['mean_drybulk_exited'],
             label="Mean Exited Drybulk Ships", marker='s')
    plateau_avg = sum(data['mean_drybulk_exited'][-4:]) / 4
    plt.axhline(y=plateau_avg, color='r', linestyle='--',
                label=f"At capacity: {plateau_avg:.2f}")
    plt.title("Mean entered vs mean exited drybulk ships")
    plt.xlabel("Arrival rate multiplier")
    plt.ylabel("Number of drybulk ships")
    plt.legend()
    plt.grid()
    plt.savefig(f"./breakpointAnalysis/{file_id}_throughput_drybulk.pdf")
    plt.close()

    # Plot 6: Channel Utilization
    means = [item[0] for item in data['channel_utilization']]
    stds = [item[1] for item in data['channel_utilization']]
    if plot_error_bars:
        plt.errorbar(arrival_factors, means, yerr=stds, fmt='o',
                     capsize=5, label="Channel Utilization")
    plt.plot(arrival_factors, means, marker='o')
    plateau_avg = sum(means[-4:]) / 4
    plt.axhline(y=plateau_avg, color='r', linestyle='--',
                label=f"At capacity: {plateau_avg:.2f}")
    plt.ylim(bottom=int(min(means)))
    plt.ylim(top=int(max(means) + 1))
    plt.title("Channel Utilization")
    plt.xlabel("Arrival rate multiplier")
    plt.ylabel("Mean Vessels in Channel (± SD)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./breakpointAnalysis/{file_id}_Channel.pdf")
    plt.close()

    output_file = f"./breakpointAnalysis/{file_id}_channel_utilization.csv"
    with open(output_file, 'w') as f:
        f.write("Arrival Rate,Mean Channel Utilization\n")
        for i in range(len(arrival_factors)):
            f.write(f"{arrival_factors[i]},{means[i]}\n")
    print(f"Data saved to {output_file}")

    categories = ["container", "liquid", "drybulk"]
    means = {cat: [item[0] for item in data[cat]] for cat in categories}
    stds = {cat: [item[1] for item in data[cat]] for cat in categories}

    # Plot 7: Anchorage Queue Lengths
    for j, category in enumerate(categories):
        if plot_error_bars:

            plt.errorbar(
                x=arrival_factors,
                y=means[category],
                yerr=stds[category],
                label=f"{category.capitalize()} Ships",
                fmt='o',
                capsize=5
            )
        plt.plot(arrival_factors, means[category], marker='o')
        plt.title(f"Anchorage queue lengths for {category} ships")
        plt.xlabel("Arrival rate multiplier")
        plt.ylabel("Queue length (mean ± SD)")
        plt.grid()
        plt.savefig(
            f"./breakpointAnalysis/{file_id}_AnchorageQueue_{category}.pdf")
        plt.close()

    # Plot 9: Anchorage queue plots for all ships togetehr
    mean_all = [np.sum([means[cat][i] for cat in categories])
                for i in range(len(arrival_factors))]
    std_all = [np.sum([stds[cat][i] for cat in categories])
               for i in range(len(arrival_factors))]

    plt.plot(arrival_factors, mean_all,
             label="Mean Anchorage Queue Lengths", marker='o')
    if plot_error_bars:
        plt.errorbar(arrival_factors, mean_all,
                     yerr=std_all, fmt='o', capsize=5)
    plt.title("Anchorage queue lengths for all ships")
    plt.xlabel("Arrival rate multiplier")
    plt.ylabel("Queue length (mean ± SD)")
    plt.legend()
    plt.grid()
    plt.savefig(f"./breakpointAnalysis/{file_id}_AnchorageQueue_all.pdf")
    plt.close()

    plt.plot(arrival_factors[:k], mean_all[:k],
             label="Mean Anchorage Queue Lengths", marker='o')
    if plot_error_bars:
        plt.errorbar(arrival_factors, mean_all,
                     yerr=std_all, fmt='o', capsize=5)
    plt.title("Anchorage queue lengths for all ships")
    plt.xlabel("Arrival rate multiplier")
    plt.ylabel("Queue length (mean ± SD)")
    plt.legend()
    plt.grid()
    plt.savefig(f"./breakpointAnalysis/{file_id}_AnchorageQueue_all_{k}.pdf")
    plt.close()

    output_file = f"./breakpointAnalysis/{file_id}_AnchorageQueue_all.csv"
    with open(output_file, 'w') as f:
        f.write("Arrival Rate,Mean Anchorage Queue Length\n")
        for i in range(len(arrival_factors)):
            f.write(f"{arrival_factors[i]},{mean_all[i]}\n")
    print(f"Data saved to {output_file}")
    print(f"Anchorage queue lengths data saved to {output_file}")


if __name__ == '__main__':

    # File ID
    file_id = constants.LOG_NUM
    inter_arrival_factors_list = [1.0] + np.arange(0.5, 5.5, 0.5).tolist()
    print(inter_arrival_factors_list)
    num_test = len(inter_arrival_factors_list)
    log_ids = [file_id + i for i in range(num_test)]


    inter_arrival_factors_ctr = inter_arrival_factors_list
    inter_arrival_factors_liq = inter_arrival_factors_list
    inter_arrival_factors_drybulk = inter_arrival_factors_list

    inter_arrival_factors = [
        (inter_arrival_factors_ctr[i], inter_arrival_factors_liq[i],
         inter_arrival_factors_drybulk[i])
        for i in range(len(inter_arrival_factors_ctr))
    ]
    param_combinations = [(log_ids[i], inter_arrival_factors[i])
                          for i in range(len(log_ids))]

    for params in param_combinations:
        print(f"Running simulation for LOG_NUM={params[0]} with inter-arrival factors {params[1]}")
        run_simulation(params)

    simulation_time = constants.SIMULATION_TIME

    # Uncomment the following lines to analyze and plot data
    base_path = "./collatedResults"
    data = extract_data(log_ids, simulation_time, base_path)
    plot_data(data, plot_error_bars=False)

    df = pd.read_csv(f"./breakpointAnalysis/{file_id}_throughput.csv")
    df.columns = ['lambda_multiplier', 'entries', 'exits']
    df['lambda'] = df['lambda_multiplier'] * constants.base_arrival_rate
    df.to_csv(f"./breakpointAnalysis/{file_id}_throughput.csv")
    df = df.drop(index=0)
    print(df)
    Cu_fit, theta_fit = fit_ODE_model(df)

    print("All simulations completed.")
