"""
Creates output logs and plots for interpreting and analyzing results.
"""

import os
import re
import math

import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm

import constants
from simulation_handler.helpers import log_line


nan = float('nan')

#######################
# Process charts
#######################


def plot_process_chart(events, run_id):
    """
    # TODO: Ensure this is compatible with the new event structure.
    Generates process charts for each terminal based on the events.
    Args:
        events (list): List of events where each event is a tuple containing:
                       (event_name, terminal_type, terminal_id, event_type, time, additional_info)
        run_id (str): Unique identifier for the run to save the plots.
    Returns:
        None: Saves process charts as images in the specified directory.
    """
    terminal_events = {}

    output_dir = f'.{run_id}/plots/TerminalProcessCharts'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for event in events:
        try:
            _, terminal_type, terminal_id, event_type, time, _ = event
            key = (terminal_type, terminal_id)
            if key not in terminal_events:
                terminal_events[key] = {}
            terminal_events[key].setdefault(event_type, []).append(time)
        except:
            if event[0] != "Truck Arrival" and event[0] != "Truck Departure":
                print(event)
            pass

    for (terminal_type, terminal_id), events in terminal_events.items():
        try:
            df = pd.DataFrame()
            df['Time'] = events['arrive'] + events['dock'] + \
                sum([events[e] for e in events if e not in [
                    'arrive', 'dock', 'depart']], []) + events['depart']
            df['Event'] = ['Ship Arrival'] * len(events['arrive']) + ['Ship Dock'] * len(events['dock']) + sum([[e] * len(
                events[e]) for e in events if e not in ['arrive', 'dock', 'depart']], []) + ['Ship Depart'] * len(events['depart'])

            plt.figure(figsize=(10, 6))
            plt.scatter(df['Time'], df['Event'])
            plt.title(
                f'Process Chart for {terminal_type} Terminal {terminal_id}')
            plt.xlabel('Time')
            plt.ylabel('Event')

            plt.savefig(
                f'{output_dir}/process_chart_{terminal_type}_{terminal_id}.jpg')
            plt.close()
        except KeyError:
            pass

#######################
# Dwell times and Turn times
#######################


def plot_dwell_times(events, run_id):
    """
    Generates plots for dwell times and turn times of ships at terminals.
    Args:
        events (list): List of events where each event is a tuple containing:
                       (ship_name, terminal_type, terminal_id, event_type, time, additional_info)
        run_id (str): Unique identifier for the run to save the plots.
    Returns:
        None: Saves plots as images in the specified directory.
    """

    all_dwell_times = []
    all_turn_times = []

    ship_arrival_times = []
    ship_dock_times = []
    ship_undock_times = []
    ship_departure_times = []

    dwell_times = {}
    turn_times = {}

    # read the events
    for event in events:
        try:
            name, terminal_type, terminal_id, event_type, time, _ = event
            if event_type == "arrive":  # arrive at anchorage
                ship_arrival_times.append(
                    (name, terminal_type, terminal_id, time))
            if event_type == "dock":
                ship_dock_times.append(
                    (name, terminal_type, terminal_id, time))
            if event_type == "undock":
                ship_undock_times.append(
                    (name, terminal_type, terminal_id, time))
            elif event_type == "depart":
                ship_departure_times.append(
                    (name, terminal_type, terminal_id, time))
        except:
            pass

    # Calculate turn times
    for ship, terminal_type, terminal_id, arrival_time in ship_arrival_times:
        for name, t_type, t_id, departure_time in ship_departure_times:
            if ship == name and terminal_type == t_type and terminal_id == t_id:
                key = (terminal_type, terminal_id)
                if key not in turn_times:
                    turn_times[key] = {}
                turn_times[key][ship] = departure_time - arrival_time
                all_turn_times.append(departure_time - arrival_time)

    # Calculate dwell times
    for ship, terminal_type, terminal_id, dock_time in ship_dock_times:
        for name, t_type, t_id, undock_time in ship_undock_times:
            if ship == name and terminal_type == t_type and terminal_id == t_id:
                key = (terminal_type, terminal_id)
                if key not in dwell_times:
                    dwell_times[key] = {}
                dwell_times[key][ship] = undock_time - dock_time
                all_dwell_times.append(undock_time - dock_time)

    # Create bar plot of the dwell times for individual vessels, by unique terminal
    for (terminal_type, terminal_id), ships_dwell_times in dwell_times.items():
        try:
            df = pd.DataFrame(ships_dwell_times.items(),
                              columns=['Ship', 'Dwell Time'])
            # Set 'Ship' as index for proper labeling on x-axis
            df.set_index('Ship', inplace=True)
            df.plot(kind='bar', y='Dwell Time', legend=False,
                    title=f'Dwell times of ships at {terminal_type} terminal {terminal_id}')
            plt.ylabel('Dwell time (hr)')
            plt.savefig(
                f'.{run_id}/plots/DwellTimes/dwell_times_{terminal_type}_{terminal_id}.jpg')
            plt.close()  # Clear the figure after saving the plot
        except Exception as e:
            print(
                f"Error in dwell time plot generation for {terminal_type} terminal {terminal_id}: {e}")
            pass

    # Create bar plot of the turn times for individual vessels, by unique terminal
    for (terminal_type, terminal_id), ships_turn_times in turn_times.items():
        try:
            df = pd.DataFrame(ships_turn_times.items(),
                              columns=['Ship', 'Turn Time'])
            # Set 'Ship' as index for proper labeling on x-axis
            df.set_index('Ship', inplace=True)
            df.plot(kind='bar', y='Turn Time', legend=False,
                    title=f'Turn Times of Ships at {terminal_type} Terminal {terminal_id}')
            plt.ylabel('Turn Time')
            plt.savefig(
                f'.{run_id}/plots/TurnTimes/turn_times_{terminal_type}_{terminal_id}.jpg')
            plt.close()  # Clear the figure after saving the plot
        except Exception as e:
            print(
                f"Error in turn time plot generation for {terminal_type} Terminal {terminal_id}: {e}")
            pass

    # Create histogram of the dwell times of all vessels visiting each unique terminal
    for (terminal_type, terminal_id), ships_dwell_times in dwell_times.items():
        try:
            dwell_times_list = list(ships_dwell_times.values())
            plt.figure(figsize=(10, 6))
            plt.hist(dwell_times_list, bins=20, edgecolor='black')
            plt.title(
                f'Dwell time distribution at {terminal_type} terminal {terminal_id}')
            plt.xlabel('Dwell time (hr)')
            plt.ylabel('Frequency')
            plt.savefig(
                f'.{run_id}/plots/DwellTimesDist/dwell_distribution_{terminal_type}_{terminal_id}.jpg')
            plt.close()
        except Exception as e:
            print(
                f"Error in dwell time distribution plot generation for {terminal_type} terminal {terminal_id}: {e}")
            pass

    # Create histogram of the turn times of all vessels visiting each unique terminal
    for (terminal_type, terminal_id), ships_turn_times in turn_times.items():
        try:
            turn_times_list = list(ships_turn_times.values())
            plt.figure(figsize=(10, 6))
            plt.hist(turn_times_list, bins=20, edgecolor='black')
            plt.title(
                f'Turn time distribution at {terminal_type} terminal {terminal_id}', fontsize=20)
            plt.xlabel('Turn time (hr)', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.savefig(
                f'.{run_id}/plots/TurnTimesDist/turn_distribution_{terminal_type}_{terminal_id}.jpg')
            plt.close()
        except Exception as e:
            print(
                f"Error in turn time distribution plot generation for {terminal_type} terminal {terminal_id}: {e}")
            pass

    # Create histogram of dwell times for all vessels in port system (all terminals)
    all_dwell_times = [time for terminal_dwell_times in dwell_times.values(
    ) for time in terminal_dwell_times.values()]
    plt.figure(figsize=(10, 6))
    plt.hist(all_dwell_times, bins=20, edgecolor='black')
    plt.title('Overall vessel terminal dwell time distribution', fontsize=20)
    plt.xlabel('Dwell time (hr)', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f'.{run_id}/plots/dwell_time_distribution.jpg')
    plt.close()

    # Create histogram of turn times for all vessels in port system (all terminals)
    all_turn_times = [time for terminal_turn_times in turn_times.values()
                      for time in terminal_turn_times.values()]
    plt.figure(figsize=(10, 6))
    plt.hist(all_turn_times, bins=20, edgecolor='black')
    plt.title('Overall vessel turn time distribution', fontsize=20)
    plt.xlabel('Turn time (hr)', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f'.{run_id}/plots/turn_time_distribution.jpg')
    plt.close()

    report_path = f".{run_id}/logs/final_report.txt"
    with open(report_path, 'a') as f:
        f.write("Dwell Time Analysis:\n")
        f.write(f"Max dwell time: {max(all_dwell_times)}\n")
        f.write(
            f"Median dwell time: {sorted(all_dwell_times)[len(all_dwell_times) // 2]}\n")
        f.write(
            f"Average dwell time: {sum(all_dwell_times) / len(all_dwell_times)}\n\n")

        f.write("Turn Time Analysis:\n")
        f.write(f"Max turn time: {max(all_turn_times)}\n")
        f.write(
            f"Median turn time: {sorted(all_turn_times)[len(all_turn_times) // 2]}\n")
        f.write(
            f"Average turn time: {sum(all_turn_times) / len(all_turn_times)}\n\n")

    return None


def analyze_turn_time_ships(run_id):
    """ Analyze the turn times and dwell times of ships from the Excel file and save the results to a report.
    Args:
        run_id (str): Unique identifier for the run to save the report.
    Returns:
        None: Saves the analysis results to a text file in the specified directory.
    """
    # Load the Excel file
    file_path = f'.{run_id}/logs/ship_logs.xlsx'

    df = pd.read_excel(file_path)
    mean_dwell_times = df.groupby(['Terminal Directed'])[
        'Turn Time'].mean().reset_index()
    with open(f".{run_id}/logs/final_report.txt", 'a') as f:
        f.write("Mean Turn Times by Terminal Directed:\n")
        f.write(str(mean_dwell_times) + "\n\n")
    mean_dwell_times = df.groupby(['Terminal Type'])[
        'Turn Time'].mean().reset_index()
    with open(f".{run_id}/logs/final_report.txt", 'a') as f:
        f.write("Mean Turn Times by Terminal Type:\n")
        f.write(str(mean_dwell_times) + "\n\n")

    mean_dwell_times = df.groupby(['Terminal Directed'])[
        'Dwell Time'].mean().reset_index()
    with open(f".{run_id}/logs/final_report.txt", 'a') as f:
        f.write("Mean Dwell Times by Terminal Directed:\n")
        f.write(str(mean_dwell_times) + "\n\n")
    mean_dwell_times = df.groupby(['Terminal Type'])[
        'Dwell Time'].mean().reset_index()
    with open(f".{run_id}/logs/final_report.txt", 'a') as f:
        f.write("Mean Dwell Times by Terminal Type:\n")
        f.write(str(mean_dwell_times) + "\n\n")


#######################
# Queues
#######################

def plot_queues(SHIPS_IN_CHANNEL_TRACK, name, place, run_id):
    """
    Plots the number of ships in the channel over time.
    Args:
        SHIPS_IN_CHANNEL_TRACK (list): List of tuples containing time and number of ships in the channel.
        name (str): Name of the ship type (e.g., "Container", "Liquid", "Dry Bulk").
        place (str): Place where the ships are tracked (e.g., "Port").
        run_id (str): Unique identifier for the run to save the plot.
    Returns:    
        None: Saves the plot as a PDF file in the specified directory.
    """

    # Plotting
    plt.figure(figsize=(12, 8))

    x = [i[0] for i in SHIPS_IN_CHANNEL_TRACK]
    y = [i[1] for i in SHIPS_IN_CHANNEL_TRACK]
    plt.plot(x, y)
    plt.suptitle(f'Number of {name} over time', fontsize=20)
    plt.title(f'at {place}', fontsize=20)
    plt.xlabel('Time', fontsize=15)
    plt.ylabel(f'Number of {name}', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.xlim(min(x)-1, max(x)+2)
    plt.ylim(min(y)-1, max(y)+2)
    plt.savefig(
        f'.{run_id}/plots/{name}in{place}OverTime{constants.mean_interarrival_time_total}.pdf')

    report_path = "./logs/final_report.txt"
    with open(report_path, 'a') as f:
        f.write(f"Mean number of {name} in {place}: {sum(y) / len(y)}\n\n")

    plt.close()


#######################
# Resources
#######################


def track_utilization(container_data, liquid_data, drybulk_data, run_id):
    """
    Calculates the mean utilization of resources (berths and yards) for container, liquid, and dry bulk terminals.
    Args:
        container_data (pd.DataFrame): DataFrame containing container terminal data.
        liquid_data (pd.DataFrame): DataFrame containing liquid terminal data.
        drybulk_data (pd.DataFrame): DataFrame containing dry bulk terminal data.
        run_id (str): Unique identifier for the run to save the results.
    Returns:
        list: A list containing the mean utilization of resources for each terminal type.
    """

    track_list = []

    excel_data = f'.{run_id}/logs/availability.xlsx'

    def calculate_mean_utilization(data, resource_type):
        available_columns = [
            col for col in data.columns if 'Available' in col and resource_type in col]
        used_columns = [
            col for col in data.columns if 'Used' in col and resource_type in col]

        utilization = {}
        for available, used in zip(available_columns, used_columns):
            terminal_name = available.split('_Available_')[0]
            utilization[terminal_name] = data[used].sum(
            ) / (data[available].sum() + data[used].sum())

        overall_utilization = sum(data[used_columns].sum(
        )) / (sum(data[available_columns].sum()) + sum(data[used_columns].sum()))

        return utilization, overall_utilization

    container_berth_utilization, container_berth_overall = calculate_mean_utilization(
        container_data, 'Berth_Ctr')
    container_yard_utilization, container_yard_overall = calculate_mean_utilization(
        container_data, 'Yard')

    liquid_berth_utilization, liquid_berth_overall = calculate_mean_utilization(
        liquid_data, 'Berth_liq')
    liquid_tank_utilization, liquid_tank_overall = calculate_mean_utilization(
        liquid_data, 'Tank')

    drybulk_berth_utilization, drybulk_berth_overall = calculate_mean_utilization(
        drybulk_data, 'Berth_db')
    drybulk_silo_utilization, drybulk_silo_overall = calculate_mean_utilization(
        drybulk_data, 'Silo')

    track_list = [container_berth_utilization, container_berth_overall, container_yard_utilization, container_yard_overall, liquid_berth_utilization,
                  liquid_berth_overall, liquid_tank_utilization, liquid_tank_overall, drybulk_berth_utilization, drybulk_berth_overall, drybulk_silo_utilization, drybulk_silo_overall]

    return track_list


def get_utilization(container_data, liquid_data, drybulk_data, run_id):
    """
    Calculates the mean utilization of resources (berths and yards) for container, liquid, and dry bulk terminals.
    Args:
        container_data (pd.DataFrame): DataFrame containing container terminal data.
        liquid_data (pd.DataFrame): DataFrame containing liquid terminal data.
        drybulk_data (pd.DataFrame): DataFrame containing dry bulk terminal data.
        run_id (str): Unique identifier for the run to save the results.
    Returns:
        list: A list containing the mean utilization of resources for each terminal type.
    """

    track_list = []

    excel_data = f'.{run_id}/logs/availability.xlsx'

    def calculate_mean_utilization(data, resource_type):
        available_columns = [
            col for col in data.columns if 'Available' in col and resource_type in col]
        used_columns = [
            col for col in data.columns if 'Used' in col and resource_type in col]
        # print("used:", used_columns)
        utilization = {}
        for available, used in zip(available_columns, used_columns):
            terminal_name = available.split('_Available_')[0]
            # print("data", data)
            # print("used:", used)
            # print(data[used])
            # print(list(data[used]))
            utilization[terminal_name] = list(
                data[used])[-1] / (list(data[available])[-1] + list(data[used])[-1])
        overall_utilization = data[used_columns].iloc[-1].sum() / (
            data[available_columns].iloc[-1].sum() + data[used_columns].iloc[-1].sum())

        return utilization, overall_utilization

    container_berth_utilization, container_berth_overall = calculate_mean_utilization(
        container_data, 'Berth_Ctr')
    container_yard_utilization, container_yard_overall = calculate_mean_utilization(
        container_data, 'Yard')

    liquid_berth_utilization, liquid_berth_overall = calculate_mean_utilization(
        liquid_data, 'Berth_liq')
    liquid_tank_utilization, liquid_tank_overall = calculate_mean_utilization(
        liquid_data, 'Tank')

    drybulk_berth_utilization, drybulk_berth_overall = calculate_mean_utilization(
        drybulk_data, 'Berth_db')
    drybulk_silo_utilization, drybulk_silo_overall = calculate_mean_utilization(
        drybulk_data, 'Silo')

    track_list = [container_berth_utilization, container_berth_overall, container_yard_utilization, container_yard_overall, liquid_berth_utilization,
                  liquid_berth_overall, liquid_tank_utilization, liquid_tank_overall, drybulk_berth_utilization, drybulk_berth_overall, drybulk_silo_utilization, drybulk_silo_overall]

    return track_list


def add_to_report(report_lines, resource, utilization, overall_utilization):
    """
    Adds the mean utilization of a resource to the report lines.
    Args:
        report_lines (list): List of report lines to which the utilization will be added.
        resource (str): Name of the resource (e.g., "Container berth", "Liquid storage").
        utilization (dict): Dictionary containing the mean utilization for each terminal.
        overall_utilization (float): Overall mean utilization for the resource.
    Returns:
        None: Appends the utilization information to the report lines.
    """

    report_lines.append(f"Mean {resource} Utilization:")
    for terminal, mean_util in utilization.items():
        report_lines.append(f"{terminal}: {mean_util:.2%}")
    report_lines.append(
        f"Overall {resource} Utilization: {overall_utilization:.2%}")
    report_lines.append("\n")


def save_track_list(run_id, timestep, track_list):
    """
    Saves the utilization track list to a text file for the given timestep.
    Args:
        run_id (str): Unique identifier for the run to save the report.
        timestep (int): The current timestep for which the report is generated.
        track_list (list): List containing the mean utilization of resources for each terminal type.
    Returns:
        None: Saves the report to a text file in the specified directory.
    """

    container_berth_utilization, container_berth_overall, container_yard_utilization, container_yard_overall, liquid_berth_utilization, liquid_berth_overall, liquid_tank_utilization, liquid_tank_overall, drybulk_berth_utilization, drybulk_berth_overall, drybulk_silo_utilization, drybulk_silo_overall = track_list
    report_lines = []

    # Container Terminals
    add_to_report(report_lines, "Container berth",
                  container_berth_utilization, container_berth_overall)
    add_to_report(report_lines, "Container storage",
                  container_yard_utilization, container_yard_overall)

    # Liquid Terminals
    add_to_report(report_lines, "Liquid berth",
                  liquid_berth_utilization, liquid_berth_overall)
    add_to_report(report_lines, "Liquid storage",
                  liquid_tank_utilization, liquid_tank_overall)

    # Dry Bulk Terminals
    add_to_report(report_lines, "Drybulk berth",
                  drybulk_berth_utilization, drybulk_berth_overall)
    add_to_report(report_lines, "Drybulk storage",
                  drybulk_silo_utilization, drybulk_silo_overall)

    # Save report to text file
    report_path = f'.{run_id}/logs/availability/{timestep}.txt'
    with open(report_path, 'a') as file:
        file.write(f"Time Step: {timestep}\n")
        file.write("\n".join(report_lines))


#######################
# Train
#######################

def gen_train_df(train_events, run_id):
    """
    Generates a DataFrame from the train events and saves it to a CSV file.
    Args:
        train_events (dict): Dictionary containing train events where keys are event names and values are lists of event data.
        run_id (str): Unique identifier for the run to save the DataFrame.
    Returns:
        pd.DataFrame: DataFrame containing the train events.
    """
    train_events_df = pd.DataFrame(train_events).T
    train_events_df.to_csv(f'.{run_id}/logs/train_logs.csv')
    return train_events_df


#######################
# Time distribution
#######################

def get_dists(run_id, plot=True):
    """
    Analyzes the ship logs to calculate the mean, min, max, and standard deviation of various time distributions for each terminal type.
    Args:
        run_id (str): Unique identifier for the run to save the analysis results.
        plot (bool): Whether to generate and save plots of the time distributions.
    Returns:
        tuple: A tuple containing the mean, min, max, and standard deviation DataFrames for each terminal type.
    """
    def extract_t(restriction_str):
        if isinstance(restriction_str, str):
            parts = restriction_str.split(", ")
            for part in parts:
                if part.startswith("T"):
                    return float(part.split(":")[1])
        return 0.0

    data = pd.read_excel(f'.{run_id}/logs/ship_logs.xlsx', sheet_name='Sheet1')

    data['Restriction In (T)'] = data['Time for Restriction In'].apply(
        extract_t)
    data['Restriction Out (T)'] = data['Time for Restriction Out'].apply(
        extract_t)

    data['Channel In'] = data['Time to Common Channel In'] + \
        data['Time to Travel Channel In']
    data['Channel Out'] = data['Time to Common Channel Out'] + \
        data['Time to Travel Channel Out']
    data['Terminal Ops'] = data['Loading Time'] + data['Unloading Time']
    data['Terminal Other'] = data['Waiting Time'] + data['Departure Time']

    columns = [
        'Time to get Berth', 'Restriction In (T)', 'Channel In', 'Time to get Pilot In',
        'Time to get Tugs In', 'Terminal Ops', 'Terminal Other', 'Restriction Out (T)',
        'Time to get Pilot Out', 'Time to get Tugs Out', 'Channel Out'
    ]

    mean_values = data.groupby('Terminal Type')[columns].mean()
    min_values = data.groupby('Terminal Type')[columns].min()
    max_values = data.groupby('Terminal Type')[columns].max()
    std_values = data.groupby('Terminal Type')[columns].std()

    error_bars = {'min': mean_values - min_values,
                  'max': max_values - mean_values}

    with open(f'.{run_id}/logs/final_report.txt', 'a') as file:
        file.write("Mean Time Distribution:\n")
        file.write(mean_values.to_string())
        file.write("\n\nMin Time Distribution:\n")
        file.write(min_values.to_string())
        file.write("\n\nMax Time Distribution:\n")
        file.write(max_values.to_string())
        file.write("\n\nStandard Deviation:\n")
        file.write(std_values.to_string())

    if plot:
        for terminal in mean_values.index:
            plt.figure(figsize=(10, 6))
            means = mean_values.loc[terminal]
            min_err = error_bars['min'].loc[terminal]
            max_err = error_bars['max'].loc[terminal]
            stds = std_values.loc[terminal]

            plt.errorbar(columns, means, yerr=[
                         min_err, max_err], fmt='-o', capsize=5)

            for i, (mean, min_e, max_e, std) in enumerate(zip(means, min_err, max_err, stds)):
                plt.annotate(
                    f'Mean: {mean:.1f}\nSD: {std:.1f}',
                    (i, mean),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center'
                )

            plt.xticks(rotation=90)
            plt.title(f'{terminal} Terminal: Mean with Min/Max and SD')
            plt.xlabel('Processes')
            plt.ylabel('Time (Units)')
            plt.tight_layout()
            # os.makedirs('/plots/TerminalProcessDist/', exist_ok=True)
            plt.savefig(
                f'.{run_id}/plots/TerminalProcessDist/{terminal}_terminal_plot.png')
            plt.close()
    return mean_values, min_values, max_values, std_values


#######################
# Channel and Anchorage
#######################


def plot_channel(run_id):
    """
    Plots the channel sections and their respective terminals.
    Args:
        run_id (str): Unique identifier for the run to save the plot.
    Returns:
        None: Saves the plot as a PDF file in the specified directory.
    """

    # Normalize the draft values for color mapping
    channel_sections = constants.CHANNEL_SECTION_DIMENSIONS
    last_section_dict = constants.LAST_SECTION_DICT
    drafts = [draft for _, _, draft, _ in channel_sections.values()]
    norm = mcolors.Normalize(vmin=min(drafts), vmax=max(drafts))
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(20, 8))
    max_width = max(width for _, width, _, _ in channel_sections.values())

    # Calculate cumulative lengths for x positions (rotated plot)
    cumulative_length = 0
    x_positions = {}
    for section, (length, width, draft, speed) in channel_sections.items():
        x_positions[section] = cumulative_length + \
            length / 2  # Middle of the section
        cumulative_length += length

    for section, (length, width, draft, speed) in channel_sections.items():
        color = cmap(norm(draft))
        y_pos = (max_width - width) / 2  # Center-align the rectangle
        ax.add_patch(plt.Rectangle(
            (x_positions[section] - length / 2, y_pos), length, width, edgecolor='black', facecolor=color))
        terminal_num = next((f'{terminal_type} Terminal : #{terminal}' for terminal_type, terminals in last_section_dict.items(
        ) for terminal, sec in terminals.items() if sec == section), None)
        annotation = f'Section {section}'
        if terminal_num:
            annotation += f'\n{terminal_num}'
        ax.text(x_positions[section], max_width / 2, annotation,
                ha='center', va='center', color='white', fontsize=8, rotation=90)

    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Depth (Draft)')

    ax.set_xlim(0, cumulative_length)
    ax.set_ylim(0, max_width + 5)
    ax.set_xlabel('Length')
    ax.set_ylabel('Breadth')
    ax.set_title('Channel Sections and Terminals')
    plt.savefig(f'.{run_id}/plots/ChannelSections.pdf')
    plt.close()


def ship_data_csv_1(run_id):
    """
    Parses the ship data from the log file and saves it to a CSV file. 
    The CSV file contains information about each ship's entry and exit times in the channel sections.
    Args:
        run_id (str): Unique identifier for the run to save the CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the parsed ship data.
    """
    ship_data = []
    file_path = f'.{run_id}/logs/ships_in_channel.txt'

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(
                r"Ship (\d+) of type (\w+) of width ([\d.]+) and draft ([\d.]+) spent from ([\d.]+) to ([\d.]+) in section (\d+) going (\w+)", line)
            if match:
                ship_data.append({
                    "ship": int(match.group(1)),
                    "ship_type": match.group(2),
                    "width": float(match.group(3)),
                    "draft": float(match.group(4)),
                    "section": int(match.group(7)),
                    "start": float(match.group(5)),
                    "end": float(match.group(6)),
                    "direction": match.group(8)
                })
            else:
                print(f"Error parsing line: {line}")

    df = pd.DataFrame(ship_data)
    df = df.sort_values(by=['ship', 'direction', 'section'])

    df.to_csv(
        f'.{run_id}/logs/channel_section_entry_exit_times.csv', index=False)

    return df


def ship_data_csv_2(run_id):
    """
    Parses the ship data from the log file and saves it to a CSV file with time spent in each section.
    Args:
        run_id (str): Unique identifier for the run to save the CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the parsed ship data with time spent in each section.
    """
    ship_data = {}
    file_path = f'.{run_id}/logs/ships_in_channel.txt'
    pattern = r"Ship (\d+) of type (\w+) of width ([\d.]+) and draft ([\d.]+) spent from ([\d.]+) to ([\d.]+) in section (\d+) going (\w+)"

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                ship_num = int(match.group(1))
                ship_dir = match.group(8)
                ship_type = match.group(2)
                section = int(match.group(7))
                time_spent = float(match.group(6)) - float(match.group(5))

                if (ship_num, ship_dir) not in ship_data:
                    ship_data[(ship_num, ship_dir)] = {}

                ship_data[(ship_num, ship_dir)][section] = time_spent
            else:
                print(f"Error parsing line: {line}")

        max_section = max(max(ship_info.keys())
                          for ship_info in ship_data.values())
        df = pd.DataFrame(ship_data).T
        df = df.reindex(columns=range(max_section + 1)).fillna(np.nan)
        df.columns = [f"Time in Sec {i}" for i in range(max_section + 1)]
        df['Total Time in Channel'] = df.sum(axis=1)
        df['Ship'] = [ship[0] for ship in df.index]
        df['Direction'] = [ship[1] for ship in df.index]
        section_columns = [f"Time in Sec {i}" for i in range(max_section + 1)]
        df = df[['Ship', 'Direction'] +
                ['Total Time in Channel'] + section_columns]
        df = df.sort_values(by=['Ship', 'Direction'])
        df.to_csv(f'.{run_id}/logs/ships_time_in_channel.csv', index=False)

        return df


def ships_in_channel_analysis(run_id, plot=True):
    """
    Analyzes the ships in the channel by reading the ship data from a text file, calculating the time spent in each section, and generating a time series of ships in the channel.
    Args:
        run_id (str): Unique identifier for the run to save the analysis results.
        plot (bool): Whether to generate and save plots of the channel utilization.
    Returns:
        None: Saves the analysis results to CSV files and generates plots if `plot` is True.
    """

    ship_data_csv_1(run_id)
    ship_data_csv_2(run_id)

    os.remove(f'.{run_id}/logs/ships_in_channel.txt')

    file_path = f'.{run_id}/logs/channel_section_entry_exit_times.csv'
    ships_data = pd.read_csv(file_path)
    ships_data['time_in_section'] = ships_data['end'] - ships_data['start']
    max_time = ships_data['end'].max()

    section_time_series = pd.DataFrame(0, index=np.arange(
        max_time)-1, columns=range(ships_data['section'].max() + 1))

    for _, row in ships_data.iterrows():
        section_time_series.loc[row['start']:row['end'], row['section']] += 1

    # save section time series to csv
    section_time_series.to_csv(f'.{run_id}/logs/section_time_series.csv')
    total_ships_in_channel_full = section_time_series.sum(axis=1)
    mean_ships_per_time_step = section_time_series.loc[constants.WARMUP_ITERS:].mean(
    )
    total_ships_in_channel = section_time_series.sum(axis=1)
    mean_total_ships_in_channel = total_ships_in_channel.mean()

    # save total ships in channel to csv
    total_ships_in_channel.to_csv(f'.{run_id}/logs/total_ships_in_channel.csv')

    if plot:
        plt.plot(total_ships_in_channel_full)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.title('Ships in channel', fontsize=15)
        plt.xlabel('Time (hr)', fontsize=15)
        plt.ylabel('Number of ships', fontsize=15)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'.{run_id}/plots/channel_utilization.pdf')
        plt.close()

    report_path = f".{run_id}/logs/final_report.txt"
    with open(report_path, 'a') as f:
        f.write(
            f"Mean number of vessels in channel: {mean_total_ships_in_channel}\n\n")
        f.write("Mean number of vessel in each section:\n")
        for section, mean_ships in mean_ships_per_time_step.items():
            f.write(f"Section {section}: {mean_ships}\n")
        f.write("\n")

    # Calculate entry and exit times for each ship
    file_path = f'.{run_id}/logs/channel_section_entry_exit_times.csv'
    ships_data = pd.read_csv(file_path)

    # Identify the first and last sections for each ship
    first_section = ships_data[ships_data['section'] == 0]
    last_sections = ships_data.groupby(['ship', 'direction'])[
        'section'].max().reset_index()

    # Merge last sections with the original dataset to get exit times
    last_section_exit = ships_data.merge(
        last_sections,
        on=['ship', 'section', 'direction'],
        how='inner'
    )

    # Get entry and exit times for each ship
    entry_exit_times = pd.merge(
        first_section[['ship', 'direction', 'start']].rename(
            columns={'start': 'entry_time_section_0'}),
        last_section_exit[['ship', 'direction', 'end']].rename(
            columns={'end': 'exit_time_last_section'}),
        on=['ship', 'direction'],
        how='outer'
    )

    # Save the entry and exit times to csv
    entry_exit_times.to_csv(
        f'.{run_id}/logs/entry_exit_times.csv', index=False)


def ships_in_anchorage_analysis(ship_type, run_id, plot=True):
    """
    Analyzes the ships in anchorage by reading the ship data from a CSV file, calculating the cumulative arrivals and channel entries, and generating a time series of ships waiting in anchorage.
    Args:
        ship_type (str): Type of ships to analyze ('all', 'Container', 'Liquid', 'DryBulk').
        run_id (str): Unique identifier for the run to save the analysis results.
        plot (bool): Whether to generate and save plots of the anchorage queue.
    Returns:
        None: Saves the analysis results to CSV files and generates plots if `plot` is True.
    """

    # time series of ships that entered anchorage
    input_ship_data = pd.read_csv(f'.{run_id}/logs/ship_data.csv')

    if ship_type == 'all':
        input_ship_data = input_ship_data
    else:
        input_ship_data = input_ship_data[input_ship_data['ship_type'] == ship_type]

    arrival_times = input_ship_data['arrival'].dropna(
    ).sort_values().reset_index(drop=True)
    cumulative_arrivals = pd.DataFrame(
        {'time_index': range(1, int(arrival_times.max()) + 1)})
    cumulative_arrivals['cumulative_arrivals'] = cumulative_arrivals['time_index'].apply(
        lambda x: (arrival_times <= x).sum()
    )

    # save the cumulative arrivals to csv
    cumulative_arrivals.to_csv(f'.{run_id}/logs/cumulative_arrivals.csv')

    # time series of ships that entered channel
    file_path = f'.{run_id}/logs/channel_section_entry_exit_times.csv'
    ships_data = pd.read_csv(file_path)

    if ship_type != 'all':
        ships_data = ships_data[ships_data['ship_type'] == ship_type]

    ships_data = ships_data[ships_data['direction'] == 'in']
    ships_data = ships_data.groupby('ship').first().reset_index()
    channel_enter_times = ships_data['start'].reset_index(drop=True)
    cumulative_channel_enter = pd.DataFrame(
        {'time_index': range(1, int(constants.SIMULATION_TIME) + 1)})
    cumulative_channel_enter['cumulative_channel_enter'] = cumulative_channel_enter['time_index'].apply(
        lambda x: (channel_enter_times <= x).sum()
    )

    # combine the two dfs
    combined = cumulative_arrivals.merge(
        cumulative_channel_enter, on='time_index', how='outer')
    combined['waiting_in_anchorage'] = combined['cumulative_arrivals'] - \
        combined['cumulative_channel_enter']

    # svae the combined dataframe to csv
    combined.to_csv(f'.{run_id}/logs/waiting_in_anchorage_{ship_type}.csv')

    # Calculate the total number of rows
    rows_in_combined = len(combined)
    mean_waiting_in_anchorage = combined.loc[constants.WARMUP_ITERS:]['waiting_in_anchorage'].mean(
    )

    # plot time series of waiting in anchorage
    if plot:
        plt.plot(combined['time_index'], combined['waiting_in_anchorage'])
        plt.title(f'Anchorage queue - {ship_type} ships', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xlabel('Time (hr)', fontsize=15)
        plt.ylabel('Number of ships', fontsize=15)
        plt.grid(True)
        plt.tight_layout()
    plt.savefig(f'.{run_id}/plots/anchorage_queue_{ship_type}.pdf')

    with open(f'.{run_id}/logs/final_report.txt', 'a') as f:
        f.write(
            f"Mean number of {ship_type} ships waiting in anchorage: {mean_waiting_in_anchorage}\n\n")

    plt.close()

    return

######################
# Trucks 
######################


def create_truck_csv(run_id):
    """
    Read truck_data.txt, parse into a DataFrame, save as CSV under ./<run_id>/truck_data.csv,
    and return the DataFrame.
    Args:
        run_id (str): Unique identifier for the run to save the truck data.
    Returns:
        pd.DataFrame: DataFrame containing the truck data with columns for truck ID, start time, dwell time, terminal ID, terminal type, and arrival time.
    """
    txt_path = f'.{run_id}/logs/truck_data.txt'
    data = []

    # only take first 100000 units of data
    count = 0
    with open(txt_path, 'r') as f:
        for line in f:
            count += 1
            if count >= 100000:
                break
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            row = {}
            for part in parts:
                key, val = part.split(':', 1)
                row[key.strip()] = val.strip()
            data.append(row)

    df = pd.DataFrame(data)
    # convert types
    df = df.astype({
        'truck_id'     : int,
        'start_time'   : float,
        'dwell_time'   : float,
        'terminal_id'  : int,
        'terminal_type': str
    })
    # compute arrival-of-day
    df['arrival_time'] = df['start_time'] % 24

    # make run directory & save CSV
    out_dir = f'.{run_id}/logs/'
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'truck_data.csv')
    df.to_csv(csv_path, index=False)

    return df


def plot_dwell_by_type(run_id):
    """
    Plots the dwell time distribution of trucks by terminal type.   
    Args:
        run_id (str): Unique identifier for the run to save the plots.
    Returns:
        None: Saves the plots as PDF files in the specified directories.
    """
    out_dir = f'.{run_id}/plots/TruckDwellByCargo'
    df = pd.read_csv(f'.{run_id}/logs/truck_data.csv')
    os.makedirs(out_dir, exist_ok=True)

    for ttype, grp in df.groupby('terminal_type'):
        plt.figure()
        grp['dwell_time'].hist(bins=30)
        plt.title(f'Dwell Time Distribution — {ttype}')
        plt.xlabel('Dwell Time')
        plt.ylabel('Frequency')
        # make the y ticks as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/len(grp)*100:.1f}%'))
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # save the plot
        plt.tight_layout()
        fname = f'dwell_distribution_{ttype}.pdf'
        plt.savefig(os.path.join(out_dir, fname))
        # plt.show()
        plt.close()


def plot_dwell_by_terminal(run_id):
    """
    Plots the dwell time distribution of trucks by terminal type and terminal ID.
    Args:
        run_id (str): Unique identifier for the run to save the plots.          
    Returns:
        None: Saves the plots as PDF files in the specified directories.
    """
    df = pd.read_csv(f'.{run_id}/logs/truck_data.csv')
    dir_term  = f'.{run_id}/plots/TruckDwellByTerminal'
    os.makedirs(dir_term,  exist_ok=True)

    for (ttype, tid), grp in df.groupby(['terminal_type','terminal_id']):
        plt.figure()
        grp['dwell_time'].hist(bins=30)
        plt.title(f'Dwell Time — {ttype} (Terminal {tid})')
        plt.xlabel('Dwell Time')
        plt.ylabel('Frequency')
        # make the y ticks as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/len(grp)*100:.1f}%'))
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        fname = f'dwell_distribution_{ttype}_{tid}.pdf'
        plt.savefig(os.path.join(dir_term,  fname))
        plt.close()


def plot_arrival_distributions(run_id):
    """
    Plots the arrival time distribution of trucks by terminal type and terminal ID.
    Args:
        run_id (str): Unique identifier for the run to save the plots.
    Returns:
        None: Saves the plots as JPEG files in the specified directories.   
    """
    df = pd.read_csv(f'.{run_id}/logs/truck_data.csv')
    dir_type  = f'.{run_id}/plots/TruckArrivalByCargo'
    dir_cargo = f'.{run_id}/plots/TruckArrivalByTerminal'
    os.makedirs(dir_type,  exist_ok=True)
    os.makedirs(dir_cargo, exist_ok=True)

    # by type
    for ttype, grp in df.groupby('terminal_type'):
        plt.figure()
        grp['arrival_time'].hist(bins=24, range=(0,24))
        plt.title(f'Arrival Time Distribution — {ttype}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Frequency')
        # make the y ticks as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/len(grp)*100:.1f}%'))
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        fname = f'arrival_distribution_{ttype}.jpg'
        plt.savefig(os.path.join(dir_type, fname))
        plt.close()

    # by terminal
    for (ttype, tid), grp in df.groupby(['terminal_type','terminal_id']):
        plt.figure()
        grp['arrival_time'].hist(bins=24, range=(0,24))
        plt.title(f'Arrival Time — {ttype} (Terminal {tid})')
        plt.xlabel('Hour of Day')
        plt.ylabel('Frequency')
        # make the y ticks as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/len(grp)*100:.1f}%'))
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        fname = f'arrival_distribution_{ttype}_{tid}.jpg'
        plt.savefig(os.path.join(dir_cargo, fname))
        plt.close()


def delete_truck_data_csv(run_id):
    """
    Delete the truck_data.csv file if it exists and save the truck data as a pickle file.
    Args:
        run_id (str): Unique identifier for the run to save the truck data as a pickle file.
    Returns:
        None: Deletes the truck_data.csv file and saves the truck data as a pickle file.
    """
    csv_path = f'.{run_id}/logs/truck_data.csv'
    pkl_path = f'.{run_id}/logs/truck_data.pkl'
    df = pd.read_csv(csv_path)
    df.to_pickle(pkl_path)




######################
# Logs and Plots
#######################

def update_ship_logs(ship_logs, ship_type, ship_id, selected_terminal, ship_start_time="nan", time_to_get_berth="nan", time_for_restriction_in="nan", time_to_get_pilot_in="nan",
                     time_to_get_tugs_in="nan", time_to_common_channel_in="nan", time_to_travel_channel_in="nan", time_to_tug_steer_in="nan", unloading_time="nan", loading_time="nan", waiting_time="nan",
                     departure_time="nan", time_to_get_pilot_out="nan", time_to_get_tugs_out="nan", time_to_tug_steer_out="nan", time_for_restriction_out='nan', time_for_uturn="nan",
                     time_to_travel_channel_out="nan", time_to_common_channel_out="nan", ship_end_time="nan"):
    """
    Updates the ship logs with the provided information. If a log with the same Ship ID exists, it updates the existing log; otherwise, it appends a new log.
    Note 1: The ship type is represented by a single character (e.g., "C" for Container, "L" for Liquid, "D" for Dry Bulk).
    Note 2: If the log already exists, it updates the existing log with the new values, ignoring 'nan' values.
    Args:
        ship_logs (list): List of existing ship logs.
        ship_type (str): Type of the ship (e.g., "C" for Container, "L" for Liquid, "D" for Dry Bulk).
        ship_id (int): Unique identifier for the ship.
        selected_terminal (str): The terminal where the ship is directed.
        ship_start_time (str): Start time of the ship's operation.
        time_to_get_berth (str): Time taken to get to the berth.
        time_for_restriction_in (str): Time for restriction in.
        time_to_get_pilot_in (str): Time taken to get the pilot in.
        time_to_get_tugs_in (str): Time taken to get tugs in.
        time_to_common_channel_in (str): Time taken to get to the common channel in.
        time_to_travel_channel_in (str): Time taken to travel the channel in.
        time_to_tug_steer_in (str): Time taken for tug steering in.
        unloading_time (str): Time taken for unloading.
        loading_time (str): Time taken for loading.
        waiting_time (str): Time spent waiting.
        departure_time (str): Departure time of the ship.
        time_to_get_pilot_out (str): Time taken to get the pilot out.
        time_to_get_tugs_out (str): Time taken to get tugs out.
        time_to_tug_steer_out (str): Time taken for tug steering out.
        time_for_restriction_out (str): Time for restriction out.
        time_for_uturn (str): Time taken for U-turn.
        time_to_travel_channel_out (str): Time taken to travel the channel out.
        time_to_common_channel_out (str): Time taken to get to the common channel out.
        ship_end_time (str): End time of the ship's operation.
    Returns:
        list: Updated ship logs with the new or modified log entry.
    """

    if ship_type == "C":
        ship_type_full = "Container"
    elif ship_type == "L":
        ship_type_full = "Liquid"
    elif ship_type == "D":
        ship_type_full = "Dry Bulk"

    # Create the log entry
    new_log = [
        f"Ship_{ship_id}",
        ship_type_full,
        f"{ship_type}.{selected_terminal}",
        ship_start_time,
        time_to_get_berth,
        time_for_restriction_in,
        time_to_get_pilot_in,
        time_to_get_tugs_in,
        time_to_common_channel_in,
        time_to_travel_channel_in,
        time_to_tug_steer_in,
        unloading_time,
        loading_time,
        waiting_time,
        departure_time,
        time_to_get_pilot_out,
        time_to_get_tugs_out,
        time_for_uturn,
        time_to_tug_steer_out,
        time_for_restriction_out,
        time_to_travel_channel_out,
        time_to_common_channel_out,
        None,
        None,
        ship_end_time
    ]

    # Check if the log with the same Ship ID exists
    for i, log in enumerate(ship_logs):
        if log[0] == f"Ship_{ship_id}":
            for j, vals in enumerate(new_log):
                if new_log[j] != 'nan':
                    ship_logs[i][j] = vals

            return ship_logs

    # If no existing log is found, append the new log
    ship_logs.append(new_log)

    return ship_logs


def analyse_bays_chassis_utilization(chassis_bays_utilization, num_terminals_list, run_id):
    """
    Generates plots for chassis and bay utilization at each terminal type.
    Args:
        chassis_bays_utilization (dict): Dictionary containing chassis and bay utilization data for each terminal type.
        num_terminals_list (list): List containing the number of terminals for each terminal type.
        run_id (str): Unique identifier for the run to save the plots.
    Returns:
        None: Saves the plots as PDF files in the specified directory.
    """
    for terminal_type in ["Container", "Liquid", "DryBulk"]:
        for terminal_id in range(1, num_terminals_list[["Container", "Liquid", "DryBulk"].index(terminal_type)] + 1):
            to_plot = chassis_bays_utilization[terminal_type][terminal_id]

            x_vals = []
            y_vals = []

            for vals in to_plot:
                x_vals.append(vals[0])
                y_vals.append(vals[1])

            plt.figure(figsize=(12, 8))
            plt.plot(x_vals, y_vals)
            plt.title(
                f'Chassis / Bay Utilization at {terminal_type} Terminal {terminal_id}')
            plt.xlabel('Time')
            plt.ylabel('Utilization')
            plt.savefig(
                f'.{run_id}/bottlenecks/chassisBays/{terminal_type}_Terminal_{terminal_id}.pdf')
            plt.close()


def gen_logs_and_plots(run_id, ship_logs, events, chassis_bays_utilization, num_terminals_list, train_events, channel_logs, channel_events, channel, animate=True):
    """
    Generates logs and plots from the provided ship logs, events, chassis and bay utilization data, train events, and channel logs.
    Saves the logs to an Excel file and generates various plots including process charts, dwell times, turn times, truck dwell times, and channel utilization.
    Also generates utilization reports for resources such as berths and yards.
    The following plots are generated:
        - Process chart for ship operations
        - Dwell time chart for ships
        - Turn time analysis for ships
        - Truck dwell times and arrival times at terminals
        - Channel utilization over time
        - Chassis and bay utilization at each terminal type
    The results are saved in the specified run directory.
    Args:
        run_id (str): Unique identifier for the run to save the logs and plots.
        ship_logs (list): List of ship logs containing information about each ship's operations.
        events (list): List of events where each event is a tuple containing:
                          (event_type, time, name, terminal_id, terminal_type)
        chassis_bays_utilization (dict): Dictionary containing chassis and bay utilization data for each terminal type.
        num_terminals_list (list): List containing the number of terminals for each terminal type.
        train_events (dict): Dictionary containing train events where keys are event names and values are lists of event data.
        channel_logs (list): List of channel logs containing information about the channel's operations.
        channel_events (list): List of channel events where each event is a tuple containing:
                          (event_type, time, name, section_id, section_type)
        channel (object): Channel object containing sections and their respective containers.
    Returns:
        None: Saves the logs and plots in the specified directory.
    """
    def extract_t(restriction_str):
        if isinstance(restriction_str, str):
            parts = restriction_str.split(", ")
            for part in parts:
                if part.startswith("T"):
                    return float(part.split(":")[1])
        return 0.0

    # Ship logs
    columns = ["Ship_Id", "Terminal Type", "Terminal Directed", "Start Time", "Time to get Berth", "Time for Restriction In", "Time to get Pilot In", "Time to get Tugs In", "Time to Common Channel In", "Time to Travel Channel In", "Time to Tug Steer In", "Unloading Time",
               "Loading Time", "Waiting Time", "Departure Time", "Time to get Pilot Out", "Time to get Tugs Out", "Time for U-Turn", "Time to Tug Steer Out", "Time for Restriction Out", "Time to Travel Channel Out", "Time to Common Channel Out", "Turn Time", "Dwell Time", "End Time"]
    df = pd.DataFrame(ship_logs, columns=columns)

    # all columns in float format of 2 decimal places
    for col in df.columns:
        if col not in ["Ship_Id", "Terminal Type", "Terminal Directed", "Time for Restriction In", "Time for Restriction Out"]:
            df[col] = df[col].astype(float).round(2)

    df['Ship_Id'] = df['Ship_Id'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values(by='Ship_Id')
    df['Ship_Id'] = "Ship_" + df['Ship_Id'].astype(str)
    df['Turn Time'] = df['End Time'] - df['Start Time']
    df['Dwell Time'] = df['Unloading Time'] + \
        df['Loading Time'] + df['Waiting Time']

    df['Restriction In (T)'] = df['Time for Restriction In'].apply(extract_t)
    df['Anchorage Time'] = df['Time to get Berth'] + \
        df['Restriction In (T)'] + df['Time to get Pilot In'] + \
        df['Time to get Tugs In']
    # if ship type is container add anchorage waiting_container as a column
    for index, row in df.iterrows():
        if row['Terminal Type'] == 'Container':
            df.at[index, 'Anchorage Time'] += constants.ANCHORAGE_WAITING_CONTAINER
        elif row['Terminal Type'] == 'Liquid':
            df.at[index, 'Anchorage Time'] += constants.ANCHORAGE_WAITING_LIQUID
        elif row['Terminal Type'] == 'Dry Bulk':
            df.at[index, 'Anchorage Time'] += constants.ANCHORAGE_WAITING_DRYBULK

    df.to_excel(f'.{run_id}/logs/ship_logs.xlsx', index=False)

    warmup_period = constants.WARMUP_ITERS / len(df)
    mean_anchorage_time = df['Anchorage Time'].sort_values(
    ).iloc[constants.WARMUP_ITERS:].mean()
    mean_anchorage_time_ctr = df[df['Terminal Type'] == 'Container']['Anchorage Time'].sort_values(
    ).iloc[int(warmup_period * len(df[df['Terminal Type'] == 'Container'])):].mean()
    mean_anchorage_time_liq = df[df['Terminal Type'] == 'Liquid']['Anchorage Time'].sort_values(
    ).iloc[int(warmup_period * len(df[df['Terminal Type'] == 'Liquid'])):].mean()
    mean_anchorage_time_db = df[df['Terminal Type'] == 'Dry Bulk']['Anchorage Time'].sort_values(
    ).iloc[int(warmup_period * len(df[df['Terminal Type'] == 'Dry Bulk'])):].mean()

    report_path = f".{run_id}/logs/final_report.txt"
    with open(report_path, 'a') as f:
        f.write(f"Mean Anchorage Time: {mean_anchorage_time}\n")
        f.write(f"Mean Anchorage Time Container: {mean_anchorage_time_ctr}\n")
        f.write(f"Mean Anchorage Time Liquid: {mean_anchorage_time_liq}\n")
        f.write(f"Mean Anchorage Time Dry Bulk: {mean_anchorage_time_db}\n\n")

    # Generate the process chart
    plot_process_chart(events, run_id=run_id)

    # Generate the dwell time chart
    plot_dwell_times(events, run_id)
    analyze_turn_time_ships(run_id)
    get_dists(run_id)

    # Plot queueus in channel
    ships_in_channel_analysis(run_id)
    ships_in_anchorage_analysis(ship_type='all', run_id=run_id)
    ships_in_anchorage_analysis(ship_type='Container', run_id=run_id)
    ships_in_anchorage_analysis(ship_type='DryBulk', run_id=run_id)
    ships_in_anchorage_analysis(ship_type='Liquid', run_id=run_id)

    # Trucks
    create_truck_csv(run_id)
    plot_dwell_by_type(run_id)
    plot_dwell_by_terminal(run_id)
    plot_arrival_distributions(run_id)
    delete_truck_data_csv(run_id)

    # Bays & Chassis Utilization
    analyse_bays_chassis_utilization(
        chassis_bays_utilization, num_terminals_list, run_id)

    # Train
    gen_train_df(train_events, run_id)

    # tugs and pilots
    tug_pilot_df = pd.read_csv(f'.{run_id}/logs/pilots_tugs_data.csv')

    # plot day anf night pilots
    plt.figure(figsize=(12, 8))
    plt.plot(tug_pilot_df['Time'],
             tug_pilot_df['Day Pilots'], label='Day Pilots')
    plt.title('Number of day pilots available')
    plt.xlabel('Time')
    plt.ylabel('Number of pilots')
    plt.ylim(0, constants.NUM_PILOTS_DAY[1])
    plt.savefig(f'.{run_id}/plots/DayPilots.pdf')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(tug_pilot_df['Time'],
             tug_pilot_df['Night Pilots'], label='Night Pilots')
    plt.title('Number of night pilots available')
    plt.xlabel('Time')
    plt.ylabel('Number of pilots')
    plt.ylim(0, constants.NUM_PILOTS_NIGHT[1])
    plt.savefig(f'.{run_id}/plots/NightPilots.pdf')
    plt.close()

    # plot tugboats
    plt.figure(figsize=(12, 8))
    plt.plot(tug_pilot_df['Time'], tug_pilot_df['Tugboats'], label='Tugboats')
    plt.title('Tugboats')
    plt.xlabel('Time')
    plt.ylabel('Number of tugboats available')
    plt.ylim(0, constants.NUM_TUGBOATS[1])
    plt.legend()
    plt.savefig(f'.{run_id}/plots/Tugboats.pdf')
    plt.close()

    # Animate the channel usage
    if animate:
        animate_channel_usage(run_id, channel)

    # print("Done!")


#######################
# Channel Utilization Animation (DEPRECATED)
########################

def animate_channel_usage(run_id, channel):
    """
    DEPRICIATED
    Animates the channel usage over time, showing width and draft utilization.
    Args:
        run_id (str): Unique identifier for the run to save the animation.
        channel (object): Channel object containing sections and their respective containers.
    Returns:
        None: Saves the animation as a GIF file in the specified directory.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    channel_sections = constants.CHANNEL_SECTION_DIMENSIONS

    x_max = len(channel.sections)
    y_max = 100

    def init():

        for i in range(1, len(channel.sections)+1):
            ax1.bar(i,  100 * channel.sections[i-1].width_containers[0].level /
                    channel.sections[i-1].width_containers[0].capacity, color='b')
            ax2.bar(i, 100 * channel.sections[i-1].draft_containers[0].level /
                    channel.sections[i-1].draft_containers[0].capacity, color='r')
        ax1.set_xlim(0, x_max)
        ax1.set_ylim(0, y_max)
        ax2.set_xlim(0, x_max)
        ax2.set_ylim(0, y_max)
        ax1.set_title('Width utilization')
        ax2.set_title('Draft utilization')
        return ax1.patches + ax2.patches

    progress_bar = tqdm(total=int(len(channel.sections[0].width_containers)/2))

    def animate(t):

        ax1.clear()
        ax2.clear()

        # print(f'Animating {t}')
        progress_bar.update(1)
        for i in range(1, len(channel.sections) + 1):
            ax1.bar(i,  100 * channel.sections[i-1].width_containers[t].level /
                    channel.sections[i-1].width_containers[t].capacity,  color='b')
            ax2.bar(i, 100 * channel.sections[i-1].draft_containers[t].level /
                    channel.sections[i-1].draft_containers[t].capacity, color='r')
            # print(t, i, channel.sections[i-1].width_containers[0].level, channel.sections[i-1].width_containers[0].capacity)

        ax1.set_xlim(0, x_max)
        ax1.set_ylim(0, y_max)
        ax2.set_xlim(0, x_max)
        ax2.set_ylim(0, y_max)
        fig.suptitle(f'Time Step: {t}', fontsize=16, fontweight='bold')
        return ax1.patches + ax2.patches

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=range(
        int(len(channel.sections[0].width_containers)/2)), interval=1, blit=False)

    # Save the animation as gif
    ani.save(f'.{run_id}/animations/channel_avilability.gif',
             writer='pillow', fps=1000)
    plt.close()
