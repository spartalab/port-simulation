"""
This module provides functions to analyze bottlenecks in a simulation run.
It includes parsing report files, collecting utilization data, plotting utilization trends,
saving mean utilization, and analyzing channel restrictions.
"""
import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import constants


def parse_report(file_path):
    """
    Parses a report file to extract the timestep and utilization data for each resource.
    Args:
        file_path (str): The path to the report file.
    Returns:
        tuple: A tuple containing the timestep (float) and a dictionary with utilization data.
               The dictionary has resource names as keys and another dictionary as values,
               which contains terminal names and their respective utilization percentages.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    timestep = float(re.search(r"Time Step: (\d+\.\d+)", lines[0]).group(1))

    utilization_data = {}
    resource = None

    for line in lines[1:]:
        if line.startswith("Mean "):
            resource = line.split("Mean ")[1].split(" Utilization")[0]
            utilization_data[resource] = {}
        elif resource and ":" in line:
            if "Overall" in line:
                utilization_data[resource]["Overall"] = float(
                    line.split(": ")[1].replace('%', '')) / 100
            else:
                terminal, utilization = line.split(": ")
                utilization_data[resource][terminal] = float(
                    utilization.replace('%', '')) / 100

    return timestep, utilization_data


def collect_data_from_reports(directory):
    """
    Collects utilization data from all report files in the specified directory.
    Args:
        directory (str): The path to the directory containing report files.
    Returns:
        dict: A dictionary where keys are resource names and values are dictionaries containing
              timesteps and utilization data for each terminal.
    """
    data = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"):
            timestep, utilization_data = parse_report(
                os.path.join(directory, filename))
            for resource, terminals in utilization_data.items():

                if resource not in data:
                    data[resource] = {'timestep': [], 'Overall': []}
                data[resource]['timestep'].append(timestep)
                for terminal, utilization in terminals.items():
                    if terminal not in data[resource]:
                        data[resource][terminal] = []
                    data[resource][terminal].append(utilization)

    return data


def save_mean_utilization(data, resource, output_dir, run_id):
    """
    Saves the mean utilization of a specific resource across all terminals to a text file.
    Args:
        data (dict): The utilization data collected from report files.
        resource (str): The name of the resource to analyze.
        output_dir (str): The directory where the output file will be saved.
        run_id (str): The run identifier for the simulation.
    Returns:
        None
    """
    mean_utilization = {terminal: np.mean(utilizations)
                        for terminal, utilizations in data[resource].items() if terminal != 'timestep'}

    resource_name = resource.split(' ')[1]
    terminal_type = resource.split(' ')[0]
    output_dir = f'.{run_id}/bottlenecks/{terminal_type}/{resource_name}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{resource_name}_utilization.txt")

    with open(output_path, 'w') as file:
        file.write(f"Mean {resource} utilization over time steps:\n")
        for terminal, mean_util in mean_utilization.items():
            file.write(f"{terminal}: {mean_util:.2%}\n")
        overall_mean = mean_utilization.get('Overall', 0)
        file.write(f"Overall {resource} utilization: {overall_mean:.2%}\n")


def plot_utilization(data, resource, run_id):
    """
    Plots the utilization of a specific resource across all terminals over time.
    Args:
        data (dict): The utilization data collected from report files.
        resource (str): The name of the resource to analyze.
        run_id (str): The run identifier for the simulation.
    Returns:
        None
    """
    plt.figure(figsize=(8, 6))

    sorted_indices = np.argsort(data[resource]['timestep'])
    sorted_timesteps = np.array(data[resource]['timestep'])[sorted_indices]

    colors = plt.cm.tab20.colors
    color_cycle = plt.cycler(color=colors)
    plt.gca().set_prop_cycle(color_cycle)

    mean = {}
    for idx, (terminal, utilization) in enumerate(data[resource].items()):
        if terminal != 'timestep':
            sorted_utilization = np.array(utilization)[sorted_indices]
            plt.plot(sorted_timesteps, sorted_utilization, label=terminal)
            mean[terminal] = round(
                float(np.mean(sorted_utilization[constants.WARMUP_ITERS:]) * 100), 1)

    items_per_line = 5
    lines = [
        ", ".join([f"{key}: {value:.2f}" for key, value in list(
            mean.items())[i:i + items_per_line]])
        for i in range(0, len(mean), items_per_line)
    ]

    formatted_mean = "\n".join(lines)

    # set x and y axis limits
    plt.ylim([0, 1.0])
    plt.xlim(left=0)

    plt.xlabel('Time step')
    plt.ylabel('Utilization')
    plt.title(f'{resource} utilization')
    plt.legend(loc='best', ncol=2)
    plt.grid(True)

    resource_name = resource.split(' ')[1]
    terminal_type = resource.split(' ')[0]
    output_dir = f'.{run_id}/bottlenecks/{terminal_type}/{resource_name}'
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/utilization.pdf')
    plt.close()


def save_utilization_thresholds(data, resource, output_dir):
    """
    Saves the percentage of time each terminal's utilization is above specified thresholds.
    Args:
        data (dict): The utilization data collected from report files.
        resource (str): The name of the resource to analyze.
        output_dir (str): The directory where the output file will be saved.
    Returns:
        tuple: A tuple containing the thresholds and a dictionary with utilization percentages for each terminal.
    """
    thresholds = [1.00, 0.90, 0.80, 0.70, 0.6, 0.5,
                  0.4, 0.3, 0.2, 0.1, 0]  # 100%, 90%, 80%, 70%
    output_path = os.path.join(
        output_dir, f"{resource}_Utilization_Thresholds.txt")
    os.makedirs(output_dir, exist_ok=True)
    utilization_percentages = {}  # To store results for plotting

    with open(output_path, 'w') as file:
        file.write(f"{resource} Utilization Thresholds:\n")

        for terminal, utilizations in data[resource].items():
            if terminal != 'timestep':  # Skip the timestep key
                file.write(f"\nTerminal: {terminal}\n")
                total_steps = len(utilizations)
                utilization_percentages[terminal] = []
                for threshold in thresholds:
                    count_above_threshold = sum(
                        1 for u in utilizations if u >= threshold)
                    percentage_above_threshold = (
                        count_above_threshold / total_steps) * 100
                    file.write(
                        f"Time at or above {threshold * 100:.0f}% utilization: {percentage_above_threshold:.2f}%\n")
                    utilization_percentages[terminal].append(
                        percentage_above_threshold)

    return thresholds, utilization_percentages  # Return for plotting


def plot_utilization_thresholds(thresholds, utilization_percentages, resource, output_dir):
    """
    Plots the percentage of time each terminal's utilization is above specified thresholds.
    Args:
        thresholds (list): The list of utilization thresholds.
        utilization_percentages (dict): A dictionary with utilization percentages for each terminal.
        resource (str): The name of the resource to analyze.
        output_dir (str): The directory where the plots will be saved.
    Returns:
        None
    """
    for terminal, percentages in utilization_percentages.items():
        plt.figure(figsize=(6, 6))
        plt.plot([t * 100 for t in thresholds], percentages,
                 marker='o', linestyle='-', color='b')
        plt.xlabel('Utilization more than (%)')
        plt.ylabel('Percentage of time (%)')
        plt.title(f'{resource} - {terminal} utilization')
        plt.grid(True)

        # start at zero axis
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        # max at 100
        plt.ylim(top=100)
        plt.xlim(right=100)

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(
            output_dir, f"{resource}_{terminal}_Utilization_Thresholds.pdf")
        plt.savefig(plot_path)
        plt.close()


def parse_restrictions(restriction_str):
    """
    Parses a string of restrictions into a dictionary.
    Args:
        restriction_str (str): A string containing restrictions in the format "B:1.0, D:2.0, DL:3.0, T:4.0".
    Returns:
        dict: A dictionary with restriction types as keys and their values as floats.
    """
    restrictions = {}
    for item in restriction_str.split(', '):
        key, value = item.split(':')
        restrictions[key] = float(value)
    return restrictions


def save_individual_restriction_plots(data_in, data_out, title, folder_name):
    """
    Saves individual restriction plots for "In" and "Out" phases.
    Args:
        data_in (pd.Series): Series containing the "In" phase data.
        data_out (pd.Series): Series containing the "Out" phase data.
        title (str): Title for the plots.
        folder_name (str): Folder name where the plots will be saved.
    Returns:
        None
    """
    os.makedirs(folder_name, exist_ok=True)

    # Custom tick labels: replace (-1, 0] with 0 and keep the rest as they are
    custom_ticks = ['0' if str(label) == '(-1, 0]' else str(label)
                    for label in data_in.index]

    # Plot for "In" restriction
    ax = data_in.plot(kind='bar', figsize=(10, 6))
    plt.title(f'{title} waiting time distribution (in)')
    plt.ylabel('Percentage')
    plt.xlabel('Waiting time (hr)')
    plt.xticks(ticks=range(len(data_in.index)),
               labels=custom_ticks, rotation=45)

    # Adding percentage labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='edge')

    plt.tight_layout()
    plt.savefig(f'{folder_name}/{title.lower().replace(" ", "_")}_in.png')
    plt.close()

    # Plot for "Out" restriction
    custom_ticks_out = ['0' if str(
        label) == '(-1, 0]' else str(label) for label in data_out.index]
    ax = data_out.plot(kind='bar', figsize=(10, 6))
    plt.title(f'{title} waiting time distribution (out)')
    plt.ylabel('Percentage')
    plt.xlabel('Waiting time (hr)')
    plt.xticks(ticks=range(len(data_out.index)),
               labels=custom_ticks_out, rotation=45)

    # Adding percentage labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='edge')

    plt.tight_layout()
    plt.savefig(f'{folder_name}/{title.lower().replace(" ", "_")}_out.png')
    plt.close()


def channel_restriction_analysis(run_id):
    """
    Analyzes channel restrictions based on the ship logs and generates histograms and plots.
    Args:
        run_id (str): The run identifier for the simulation.
    Returns:
        None
    """
    # Load the dataset
    df = pd.read_excel(f'.{run_id}/logs/ship_logs.xlsx')
    os.makedirs(f'.{run_id}/bottlenecks/Waterway', exist_ok=True)

    # Filtering rows where restriction values are strings
    df_filtered = df[(df['Time for Restriction In'].apply(lambda x: isinstance(x, str))) &
                     (df['Time for Restriction Out'].apply(lambda x: isinstance(x, str)))].copy()

    # Applying the parsing function to the filtered dataframe
    df_filtered.loc[:, 'Restrictions_In'] = df_filtered['Time for Restriction In'].apply(
        parse_restrictions)
    df_filtered.loc[:, 'Restrictions_Out'] = df_filtered['Time for Restriction Out'].apply(
        parse_restrictions)

    # Extracting the individual components for analysis
    restriction_in_df_filtered = pd.json_normalize(
        df_filtered['Restrictions_In'])
    restriction_out_df_filtered = pd.json_normalize(
        df_filtered['Restrictions_Out'])

    # Remove the "Q" column from both the filtered DataFrames if present
    restriction_in_df_filtered = restriction_in_df_filtered.drop(
        columns=['Q'], errors='ignore')
    restriction_out_df_filtered = restriction_out_df_filtered.drop(
        columns=['Q'], errors='ignore')

    # Creating histograms for each restriction category in the "in" and "out" phases, including "T"
    bins = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    beam_in_hist, _ = pd.cut(
        restriction_in_df_filtered['B'], bins=bins, right=True, retbins=True)
    draft_in_hist, _ = pd.cut(
        restriction_in_df_filtered['D'], bins=bins, right=True, retbins=True)
    daylight_in_hist, _ = pd.cut(
        restriction_in_df_filtered['DL'], bins=bins, right=True, retbins=True)
    total_in_hist, _ = pd.cut(
        restriction_in_df_filtered['T'], bins=bins, right=True, retbins=True)

    beam_out_hist, _ = pd.cut(
        restriction_out_df_filtered['B'], bins=bins, right=True, retbins=True)
    draft_out_hist, _ = pd.cut(
        restriction_out_df_filtered['D'], bins=bins, right=True, retbins=True)
    daylight_out_hist, _ = pd.cut(
        restriction_out_df_filtered['DL'], bins=bins, right=True, retbins=True)
    total_out_hist, _ = pd.cut(
        restriction_out_df_filtered['T'], bins=bins, right=True, retbins=True)

    # Creating dataframes for each restriction category's counts, including "T"
    in_hist_df = pd.DataFrame({
        'Beam In': beam_in_hist.value_counts().sort_index(),
        'Draft In': draft_in_hist.value_counts().sort_index(),
        'Daylight In': daylight_in_hist.value_counts().sort_index(),
        'Total In': total_in_hist.value_counts().sort_index()
    })

    out_hist_df = pd.DataFrame({
        'Beam Out': beam_out_hist.value_counts().sort_index(),
        'Draft Out': draft_out_hist.value_counts().sort_index(),
        'Daylight Out': daylight_out_hist.value_counts().sort_index(),
        'Total Out': total_out_hist.value_counts().sort_index()
    })

    # Convert counts to percentages
    total_in = in_hist_df.sum()
    total_out = out_hist_df.sum()
    in_hist_percentage_df = (in_hist_df / total_in) * 100
    out_hist_percentage_df = (out_hist_df / total_out) * 100

    # Save analysis to a text file, including "T"
    output_text_file = f'.{run_id}/bottlenecks/Waterway/restriction_simulation_analysis.txt'

    analysis_text = f"""
    Percentage distributions of the waiting times for each restriction category across hourly bins, including Total:

    **In Phase (Percentages):**
    {in_hist_percentage_df.to_string()}

    **Out Phase (Percentages):**
    {out_hist_percentage_df.to_string()}
    """

    with open(output_text_file, 'w') as file:
        file.write(analysis_text)

    # save two dataframes to csv
    in_hist_percentage_df.to_csv(
        f'.{run_id}/bottlenecks/Waterway/in_hist_percentage.csv')
    out_hist_percentage_df.to_csv(
        f'.{run_id}/bottlenecks/Waterway/out_hist_percentage.csv')

    # Saving plots for each restriction separately, including "T"
    save_individual_restriction_plots(
        in_hist_percentage_df['Beam In'], out_hist_percentage_df['Beam Out'], 'Beam', f'.{run_id}/bottlenecks/Waterway/beam')
    save_individual_restriction_plots(
        in_hist_percentage_df['Draft In'], out_hist_percentage_df['Draft Out'], 'Draft', f'.{run_id}/bottlenecks/Waterway/draft')
    save_individual_restriction_plots(
        in_hist_percentage_df['Daylight In'], out_hist_percentage_df['Daylight Out'], 'Daylight', f'.{run_id}/bottlenecks/Waterway/daylight')
    save_individual_restriction_plots(
        in_hist_percentage_df['Total In'], out_hist_percentage_df['Total Out'], 'Total', f'.{run_id}/bottlenecks/Waterway/total')


def terminal_analysis(run_id):
    """
    Analyzes terminal utilization data and generates plots and reports.
    Args:
        run_id (str): The run identifier for the simulation.
    Returns:
        None
    """

    data = collect_data_from_reports(directory=f'.{run_id}/logs/availability/')
    for resource in data.keys():
        plot_utilization(data, resource, run_id)
        save_mean_utilization(
            data, resource, output_dir='./logs/', run_id=run_id)
        resource_name = resource.split(' ')[1]
        terminal_type = resource.split(' ')[0]
        thresholds, utilization_percentages = save_utilization_thresholds(
            data, resource, output_dir=f'.{run_id}/bottlenecks/{terminal_type}/{resource_name}')
        plot_utilization_thresholds(thresholds, utilization_percentages, resource,
                                    output_dir=f'.{run_id}/bottlenecks/{terminal_type}/{resource_name}/thresholds')


def bottleneckAnalysis(run_id):
    """
    Main function to run the bottleneck simulation_analysis.
    Args:
        run_id (str): The run identifier for the simulation.
    Returns:
        None
    """
    terminal_analysis(run_id)
    channel_restriction_analysis(run_id)
