"""
This module contains functions to model disruptions in a port simulation environment.
"""
import random
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from simulation_classes.port import Crane, Pipeline, Conveyor
from simulation_handler.helpers import get_value_by_terminal
from constants import SEVERE_HURRICANE, TROPICAL_STORM, HURRICANE_START, WARMUP_ITERS, NUM_MONTHS, NUM_RUNS, LOG_NUM, INBOUND_CLOSED, OUTBOUND_CLOSED




def reduce_cranes(env, change, terminals, time_start, time_end, port_berths_container_terminals, terminal_data_cache, berths_aff_per):
    """
    This function reduces the number of cranes in each berth by `change` amount by removing cranes from the berth's crane store between `time_start` and `time_end`.
    It also restores the cranes after the reduction period.
    Args:
        env (simpy.Environment): The simulation environment.
        change (int): The number of cranes to be removed (negative value) or added (positive value).
        terminals (list): List of terminal indices where the cranes will be reduced.
        time_start (int): The time at which the reduction starts.
        time_end (int): The time at which the reduction ends.
        port_berths_container_terminals (dict): Dictionary containing berth information for container terminals.
        terminal_data_cache (dict): Cache containing terminal data for crane transfer rates.
        berths_aff_per (float): Percentage of berths to be affected by the change.
    Returns:
        None
    """
    port_berth_initially = {}
    for terminal_idx in terminals:
        port_berth_initially[terminal_idx] = []
        for berths in port_berths_container_terminals[terminal_idx].items:
            port_berth_initially[terminal_idx].append(berths)

    print(
        f"Modelling disruption: Attempting to remove {-change} cranes from each berth in terminals {terminals} from time {time_start} to {time_end}")

    yield env.timeout(time_start)

    for terminal_idx in terminals:
        num_berths = len(port_berth_initially[terminal_idx])
        berths_aff = int(berths_aff_per*num_berths)
        count = 0
        for berth in port_berth_initially[terminal_idx]:
            count += 1
            if change < 0:
                num_to_remove = min(len(berth.cranes.items), abs(change))
                for _ in range(num_to_remove):
                    if berth.cranes.items:
                        crane = yield berth.cranes.get()
            if count >= berths_aff:
                break

    yield env.timeout(time_end - time_start)

    for terminal_idx in terminals:
        num_berths = len(port_berth_initially[terminal_idx])
        berths_aff = int(berths_aff_per*num_berths)
        count = 0
        for berth in port_berth_initially[terminal_idx]:
            count += 1
            if change < 0:
                for i in range(abs(change)):
                    restored_crane = Crane(id=f"restored_{berth.id}_{i}", width="width", crane_transfer_rate=get_value_by_terminal(
                        terminal_data_cache, 'Container', terminal_idx+1, 'transfer rate per unit'))
                    yield berth.cranes.put(restored_crane)
            if count >= berths_aff:
                break


def reduce_pipelines(env, change, terminals, time_start, time_end, port_berth_liquid_terminals, terminal_data_cache, berths_aff_per):
    """
    This function reduces the number of pipelines in each berth by `change` amount by removing pipelines from the berth's pipeline store between `time_start` and `time_end`.
    It also restores the pipelines after the reduction period.
    Args:
        env (simpy.Environment): The simulation environment.
        change (int): The number of pipelines to be removed (negative value) or added (positive value).
        terminals (list): List of terminal indices where the pipelines will be reduced.
        time_start (int): The time at which the reduction starts.
        time_end (int): The time at which the reduction ends.
        port_berth_liquid_terminals (dict): Dictionary containing berth information for liquid terminals.
        terminal_data_cache (dict): Cache containing terminal data for pipeline transfer rates.
        berths_aff_per (float): Percentage of berths to be affected by the change.
    Returns:
        None
    """
    port_berth_initially = {}
    for terminal_ids in terminals:
        terminal_idx = terminal_ids-1
        port_berth_initially[terminal_idx] = []
        for berths in port_berth_liquid_terminals[terminal_idx].items:
            port_berth_initially[terminal_idx].append(berths)

    print(
        f"Modelling disruption: Attempting to remove {-change} pipelines from each berth in terminals {terminals} from time {time_start} to {time_end}")

    yield env.timeout(time_start)

    for terminal_ids in terminals:
        terminal_idx = terminal_ids-1
        num_berths = len(port_berth_initially[terminal_idx])
        berths_aff = int(berths_aff_per*num_berths)
        count = 0
        for berth in port_berth_initially[terminal_idx]:
            if change < 0:
                num_to_remove = min(len(berth.pipelines.items), abs(change))
                for _ in range(num_to_remove):
                    if berth.pipelines.items:
                        pipeline = yield berth.pipelines.get()
            count += 1
            if count >= berths_aff:
                break

    yield env.timeout(time_end - time_start)

    for terminal_ids in terminals:
        terminal_idx = terminal_ids-1
        num_berths = len(port_berth_initially[terminal_idx])
        berths_aff = int(berths_aff_per*num_berths)
        count = 0
        for berth in port_berth_initially[terminal_idx]:
            count += 1
            if change < 0:
                for i in range(abs(change)):
                    restored_pipeline = Pipeline(env=env, id=f"restored_{berth.id}_{i}", pump_rate_per_pipeline=get_value_by_terminal(
                        terminal_data_cache, "Liquid", terminal_ids+1, "transfer rate per unit"))

                    yield berth.pipelines.put(restored_pipeline)
            if count >= berths_aff:
                break


def reduce_conveyors(env, change, terminals, time_start, time_end, port_berth_drybulk_terminals, terminal_data_cache, berths_aff_per):
    """
    This function reduces the number of conveyors in each berth by `change` amount by removing conveyors from the berth's conveyor store between `time_start` and `time_end`.
    It also restores the conveyors after the reduction period.
    Args:
        env (simpy.Environment): The simulation environment.
        change (int): The number of conveyors to be removed (negative value) or added (positive value).
        terminals (list): List of terminal indices where the conveyors will be reduced.
        time_start (int): The time at which the reduction starts.
        time_end (int): The time at which the reduction ends.   
        port_berth_drybulk_terminals (dict): Dictionary containing berth information for dry bulk terminals.
        terminal_data_cache (dict): Cache containing terminal data for conveyor transfer rates.
        berths_aff_per (float): Percentage of berths to be affected by the change.
    Returns:
        None
    """
    port_berth_initially = {}
    for terminal_idx in terminals:
        port_berth_initially[terminal_idx] = []
        for berths in port_berth_drybulk_terminals[terminal_idx].items:
            port_berth_initially[terminal_idx].append(berths)

    print(
        f"Modelling disruption: Attempting to remove {-change} conveyor from each berth in terminals {terminals} from time {time_start} to {time_end}")

    yield env.timeout(time_start)

    for terminal_idx in terminals:
        num_berths = len(port_berth_initially[terminal_idx])
        berths_aff = int(berths_aff_per*num_berths)
        count = 0
        for berth in port_berth_initially[terminal_idx]:
            count += 1
            if change < 0:
                num_to_remove = min(len(berth.conveyors.items), abs(change))
                for _ in range(num_to_remove):
                    if berth.conveyors.items:
                        conveyor = yield berth.conveyors.get()
            if count >= berths_aff:
                break

    yield env.timeout(time_end - time_start)

    for terminal_idx in terminals:
        num_berths = len(port_berth_initially[terminal_idx])
        berths_aff = int(berths_aff_per*num_berths)
        count = 0

        for berth in port_berth_initially[terminal_idx]:
            count += 1
            if change < 0:
                for i in range(abs(change)):
                    restored_conveyor = Conveyor(env=env, id=f"restored_{berth.id}_{i}", conveyor_rate_per_conveyor=get_value_by_terminal(
                        terminal_data_cache, "DryBulk", terminal_idx+1, "transfer rate per unit"))
                    yield berth.conveyors.put(restored_conveyor)
            if count >= berths_aff:
                break


def reduce_berths(env, change, terminals, time_start, time_end, port_berths_terminal):
    """
    This function reduces the number of berths in each terminal by `change` amount by removing berths from the terminal's berth store between `time_start` and `time_end`.
    It also restores the berths after the reduction period.
    Args:
        env (simpy.Environment): The simulation environment.
        change (int): The number of berths to be removed (negative value) or added (positive value).
        terminals (list): List of terminal indices where the berths will be reduced.        
        time_start (int): The time at which the reduction starts.
        time_end (int): The time at which the reduction ends.
        port_berths_terminal (dict): Dictionary containing berth information for the terminals.
    Returns:
        None
    """
    print(
        f"Modelling disruption: Attempting to remove {-change} berths from terminals {terminals} from time {time_start} to {time_end}")

    yield env.timeout(time_start)

    removed_berths = {terminal: [] for terminal in terminals}

    # Reduce capacity by taking items out of the store
    for terminal in terminals:
        for _ in range(abs(change)):
            # Remove a berth to simulate reduced capacity
            berth = yield port_berths_terminal[terminal].get()
            removed_berths[terminal].append(berth)
        print(
            f"Terminal {terminal}: Berth capacity effectively reduced by {abs(change)} at time {env.now}")

    yield env.timeout(time_end - time_start)

    # Restore the removed berths to the store
    for terminal in terminals:
        for berth in removed_berths[terminal]:
            yield port_berths_terminal[terminal].put(berth)
        print(
            f"Terminal {terminal}: Berth Capacity restored by {abs(change)} at time {env.now}")


def reduce_yard(env, change, terminals, time_start, time_end, port_yard_container_terminals):
    """
    This function reduces the yard capacity in each terminal by `change` amount by adding dummy containers to the terminal's yard store between `time_start` and `time_end`.
    It also restores the yard capacity after the reduction period.
    Args:
        env (simpy.Environment): The simulation environment.
        change (int): The number of yard spaces to be removed (negative value) or added (positive value).
        terminals (list): List of terminal indices where the yard capacity will be reduced.
        time_start (int): The time at which the reduction starts.
        time_end (int): The time at which the reduction ends.
        port_yard_container_terminals (dict): Dictionary containing yard information for the terminals.
    Returns:
        None"""

    print(
        f"Modelling disruption: Attempting to remove {-change} units yard space from terminals {terminals} from time {time_start} to {time_end} (Adding dummy containers)")

    yield env.timeout(time_start)

    dummy_containers = {terminal: [] for terminal in terminals}

    for terminal in terminals:
        for _ in range(abs(change)):
            ctr = yield port_yard_container_terminals[terminal].put()
            dummy_containers[terminal].append(ctr)
        print(
            f"Terminal {terminal}: Yard capacity effectively reduced by {abs(change)} ctrs at time {env.now}")

    yield env.timeout(time_end - time_start)

    # Restore the removed berths to the store
    for terminal in terminals:
        for berth in dummy_containers[terminal]:
            yield dummy_containers[terminal].get(berth)
        print(
            f"Terminal {terminal}: Yard apacity restored by {abs(change)} at time {env.now}")


def reduce_tank_capacity(env, change, terminals, time_start, time_end, port_tanks_liquid_terminals):
    """
    This function reduces the tank capacity in each terminal by `change` amount by adding dummy liquid to the terminal's tank store between `time_start` and `time_end`.
    It also restores the tank capacity after the reduction period.
    Args:
        env (simpy.Environment): The simulation environment.
        change (int): The number of tank spaces to be removed (negative value) or added (positive value).
        terminals (list): List of terminal indices where the tank capacity will be reduced.
        time_start (int): The time at which the reduction starts.
        time_end (int): The time at which the reduction ends.
        port_tanks_liquid_terminals (dict): Dictionary containing tank information for the terminals.
    Returns:
        None
    """
    print(
        f"Modelling disruption: Attempting to remove {-change} units tank space from terminals {terminals} from time {time_start} to {time_end} (Adding dummy liquid)")

    yield env.timeout(time_start)

    # Reduce the content of the tanks
    for terminal in terminals:
        yield port_tanks_liquid_terminals[terminal].put(change)
        print(
            f"Terminal {terminal}: Tank capacity effectively reduced by {abs(change)} at time {env.now}")

    yield env.timeout(time_end - time_start)

    # Restore the content of the tanks
    for terminal in terminals:
        yield port_tanks_liquid_terminals[terminal].get(change)
        print(
            f"Terminal {terminal}: Tank capacity effectively increased by {abs(change)} at time {env.now}")


def reduce_silo_capacity(env, change, terminals, time_start, time_end, port_silos_drybulk_terminals):
    """
    This function reduces the silo capacity in each terminal by `change` amount by adding dummy dry bulk to the terminal's silo store between `time_start` and `time_end`.
    It also restores the silo capacity after the reduction period.
    Args:
        env (simpy.Environment): The simulation environment.
        change (int): The number of silo spaces to be removed (negative value) or added (positive value).
        terminals (list): List of terminal indices where the silo capacity will be reduced.
        time_start (int): The time at which the reduction starts.
        time_end (int): The time at which the reduction ends.
        port_silos_drybulk_terminals (dict): Dictionary containing silo information for the terminals.
    Returns:
        None
    """
    print(
        f"Modelling disruption: Attempting to remove {-change} units silo space from terminals {terminals} from time {time_start} to {time_end} (Adding dummy dry bulk)")

    yield env.timeout(time_start)

    # Reduce the content of the silos
    for terminal in terminals:
        yield port_silos_drybulk_terminals[terminal].put(change)
        print(
            f"Terminal {terminal}: Silo capacity effectively reduced by {abs(change)} at time {env.now}")

    yield env.timeout(time_end - time_start)

    # Restore the content of the silos
    for terminal in terminals:
        yield port_silos_drybulk_terminals[terminal].get(change)
        print(
            f"Terminal {terminal}: Silo capacity effectively increased by {abs(change)} at time {env.now}")


def reduce_bay_capacity(env, reduction, terminals, start_time, end_time, port_loading_bays_liquid_terminals):
    """
    This function reduces the bay capacity in each terminal by `reduction` amount by holding requests for bays in the terminal's bay store between `start_time` and `end_time`.
    It also restores the bay capacity after the reduction period.
    Args:
        env (simpy.Environment): The simulation environment.
        reduction (int): The number of bays to be removed (negative value) or added (positive value).
        terminals (list): List of terminal indices where the bay capacity will be reduced.
        start_time (int): The time at which the reduction starts.
        end_time (int): The time at which the reduction ends.
        port_loading_bays_liquid_terminals (dict): Dictionary containing bay information for the terminals.
    Returns:
        None
    """

    print(
        f"Modelling disruption: Attempting to remove {-reduction} bays from terminals {terminals} from time {start_time} to {end_time}")

    yield env.timeout(start_time)

    # Request `reduction` number of bays to reduce the effective capacity
    requests = []
    for terminal in terminals:
        for _ in range(reduction):
            request = port_loading_bays_liquid_terminals[terminal].request()
            yield request  # Hold this request to reduce availability
            requests.append((terminal, request))
        print(
            f"{reduction} units of bays removed from terminal {terminal} at {start_time}")
    yield env.timeout(end_time - start_time)

    # Release the requests to restore full capacity
    for terminal, request in requests:
        port_loading_bays_liquid_terminals[terminal].release(request)
        print(
            f"{reduction} units of bays restored from terminal {terminal} at {end_time}")


def reduce_chassis(env, change, terminals, time_start, time_end, port_chassis_container_terminals):
    """
    This function reduces the chassis capacity in each terminal by `change` amount by removing chassis from the terminal's chassis store between `time_start` and `time_end`.
    It also restores the chassis capacity after the reduction period.
    Args:
        env (simpy.Environment): The simulation environment.
        change (int): The number of chassis to be removed (negative value) or added (positive value).
        terminals (list): List of terminal indices where the chassis capacity will be reduced.
        time_start (int): The time at which the reduction starts.
        time_end (int): The time at which the reduction ends.
        port_chassis_container_terminals (dict): Dictionary containing chassis information for the terminals.
    Returns:
        None
    """

    print(
        f"Modelling disruption: Attempting to remove {-change} chassis from terminals {terminals} from time {time_start} to {time_end}")

    yield env.timeout(time_start)

    # Reduce the content of the silos
    for terminal in terminals:
        yield port_chassis_container_terminals[terminal].get(change)
        print(
            f"Terminal {terminal}: Number of chassis reduced by {abs(change)} at time {env.now}")

    yield env.timeout(time_end - time_start)

    # Restore the content of the silos
    for terminal in terminals:
        yield port_chassis_container_terminals[terminal].put(change)
        print(
            f"Terminal {terminal}: Number of chassis increased by {abs(change)} at time {env.now}")


def reduce_truck_gates(env, reduction, terminals, start_time, end_time, truck_gates):
    """
    This function reduces the truck gate capacity in each terminal by `reduction` amount by holding requests for truck gates in the terminal's truck gate store between `start_time` and `end_time`.
    It also restores the truck gate capacity after the reduction period.
    Args:
        env (simpy.Environment): The simulation environment.
        reduction (int): The number of truck gates to be removed (negative value) or added (positive value).
        terminals (list): List of terminal indices where the truck gate capacity will be reduced.
        start_time (int): The time at which the reduction starts.
        end_time (int): The time at which the reduction ends.
        truck_gates (dict): Dictionary containing truck gate information for the terminals.
    Returns:
        None
    """
    print(
        f"Modelling disruption: Attempting to remove {-reduction} truck gates from terminals {terminals} from time {start_time} to {end_time}")

    yield env.timeout(start_time)

    # Request `reduction` number of bays to reduce the effective capacity
    requests = []
    for terminal in terminals:
        for _ in range(reduction):
            request = truck_gates[terminal].request()
            yield request  # Hold this request to reduce availability
            requests.append((terminal, request))
        print(
            f"{reduction} truck gates removed from terminal {terminal} at {start_time}")
    yield env.timeout(end_time - start_time)

    # Release the requests to restore full capacity
    for terminal, request in requests:
        truck_gates[terminal].release(request)
        print(
            f"{reduction} truck gates restored from terminal {terminal} at {end_time}")


def adjust_arrival_rate(start_time, end_time, rate_parameter, run_id, plot_input=True):
    """
    This function adjusts the arrival times of ships in the ship data CSV file by a given rate parameter.
    It modifies the arrival times of ships that fall within the specified time range and saves the updated data back to the CSV file.
    Args:
        start_time (float): The start time of the range to adjust arrival times.
        end_time (float): The end time of the range to adjust arrival times.
        rate_parameter (float): The factor by which to adjust the arrival times.
        run_id (str): The unique identifier for the simulation run.
        plot_input (bool): Whether to plot the input data before and after adjustment.
    returns:
        ship_data_df (pd.DataFrame): The updated ship data DataFrame with adjusted arrival times.
        ship_data (dict): The updated ship data as a dictionary.
    """

    ship_data = pd.read_csv(f".{run_id}/logs/ship_data.csv")
    in_time_range = (ship_data['arrival'] >= start_time) & (
        ship_data['arrival'] <= end_time)
    ship_data['old_arrival'] = ship_data['arrival']
    ship_data.loc[in_time_range, 'arrival'] /= rate_parameter

    ship_data_df = ship_data.drop(columns=['interarrival'])
    ship_data_df = ship_data_df.sort_values('arrival')
    output_path = f'.{run_id}/logs/ship_data.csv'
    ship_data_df.to_csv(output_path, index=False)
    ship_data = ship_data_df.to_dict(orient="index")

    if plot_input:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].hist(ship_data_df['old_arrival'],
                     bins=30, color='blue', alpha=0.7)
        axes[0].set_xlabel('Arrival Time')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Old Arrival Times')
        axes[1].hist(ship_data_df['arrival'], bins=30,
                     color='green', alpha=0.7)
        axes[1].set_xlabel('Arrival Time')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('New Arrival Times')
        min_x = min(ship_data_df['old_arrival'].min(),
                    ship_data_df['arrival'].min())
        max_x = max(ship_data_df['old_arrival'].max(),
                    ship_data_df['arrival'].max())
        min_y = 0
        max_y = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])

        for ax in axes:
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)

        plt.tight_layout()
        plt.savefig(f".{run_id}/plots/scenarios/factor_arrival_scnario_input")
        plt.close()

    return ship_data_df, ship_data


def stop_and_bulk_arrival(start_time, end_time, run_id, plot_input=False):
    """
    This function modifies the ship arrival times in the ship data CSV file by stopping arrivals between `start_time` and `end_time`.
    It also simulates a bulk arrival scenario after `end_time` by distributing the arrivals evenly over a recovery period.
    Args:
        start_time (float): The start time of the range to stop arrivals.
        end_time (float): The end time of the range to stop arrivals.
        run_id (str): The unique identifier for the simulation run.
        plot_input (bool): Whether to plot the input data before and after adjustment.
    returns:
        ship_data_df (pd.DataFrame): The updated ship data DataFrame with adjusted arrival times.
        ship_data_dict (dict): The updated ship data as a dictionary.
    """

    ship_data = pd.read_csv(f'.{run_id}/logs/ship_data.csv')
    ship_data['old_arrival'] = ship_data['arrival']
    in_time_range = (ship_data['arrival'] >= start_time) & (
        ship_data['arrival'] <= end_time)

    num_ships = in_time_range.sum()
    recovery_period = 24*5  # 5 days
    gap = recovery_period/num_ships
    print(
        f"Modelling disruption: Stopping arrivals between {start_time} and {end_time} and bulk arrivals after {end_time} with a gap of {gap} hours")

    bulk_arrival_time = end_time + 0.1
    ship_data.loc[in_time_range, 'arrival'] = [
        bulk_arrival_time + i * gap for i in range(in_time_range.sum())]
    if 'interarrival' in ship_data.columns:
        ship_data = ship_data.drop(columns=['interarrival'])
    ship_data_df = ship_data.sort_values('arrival')
    output_path = f'.{run_id}/logs/ship_data.csv'
    ship_data_df.to_csv(output_path, index=False)
    ship_data_dict = ship_data_df.to_dict(orient="index")

    if plot_input:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].hist(ship_data_df['old_arrival'],
                     bins=30, color='blue', alpha=0.7)
        axes[0].set_xlabel('Arrival Time')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Old Arrival Times')
        axes[1].hist(ship_data_df['arrival'], bins=30,
                     color='green', alpha=0.7)
        axes[1].set_xlabel('Arrival Time')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('New Arrival Times')
        min_x = min(ship_data_df['old_arrival'].min(),
                    ship_data_df['arrival'].min())
        max_x = max(ship_data_df['old_arrival'].max(),
                    ship_data_df['arrival'].max())
        min_y = 0
        max_y = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])

        for ax in axes:
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)

        plt.tight_layout()
        plt.savefig(f".{run_id}/plots/scenarios/bulk_arrival_scenario_input")
        plt.close()

    return ship_data_df, ship_data_dict


def filter_data_by_arrival_rate(data, time_arrival_dict, time_column):
    """
    Filter data based on arrival rate factors for different time periods.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe to filter (truck_data or train_data)
    time_arrival_dict : dict
        Dictionary with (start_time, end_time): arrival_rate_factor
        e.g., {(6.0, 10.0): 0.8, (10.0, 15.0): 0.6}
        arrival_rate_factor of 0.8 means keep 80% of data (remove 20%)
    time_column : str
        Name of the time column to use for filtering
        ('arrival_time' for trucks, 'arrival at' for trains)
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe with specified removal rates applied
    """
    
    # Create a copy to avoid modifying original data
    filtered_data = data.copy()
    
    # Track which rows to keep
    # keep_mask = pd.Series([True] * len(filtered_data))

    keep_mask = pd.Series(True, index=filtered_data.index)

    
    for (start_time, end_time), arrival_rate_factor in time_arrival_dict.items():
        # Find rows in this time window
        time_mask = (filtered_data[time_column] >= start_time) & (filtered_data[time_column] < end_time)
        rows_in_window = filtered_data[time_mask]
        
        if len(rows_in_window) == 0:
            continue
        
        # Calculate how many rows to remove
        removal_rate = 1 - arrival_rate_factor
        num_to_remove = int(len(rows_in_window) * removal_rate)
        
        if num_to_remove > 0:
            # Get indices of rows in this window
            window_indices = rows_in_window.index.tolist()
            
            # Create equispaced removal indices
            # We remove equispaced samples to maintain temporal distribution
            removal_step = len(window_indices) / num_to_remove if num_to_remove > 0 else len(window_indices)
            removal_indices = []
            
            for i in range(num_to_remove):
                idx_position = int(i * removal_step)
                if idx_position < len(window_indices):
                    removal_indices.append(window_indices[idx_position])
            
            # Mark these indices for removal
            keep_mask[removal_indices] = False
    
    # Apply the filter
    filtered_data = filtered_data[keep_mask]
    
    return filtered_data.reset_index(drop=True)

def filter_truck_data(run_id, time_arrival_dict):
    """
    Filter truck data based on arrival rate factors.
    
    Parameters:
    -----------
    run_id : str
        Unique identifier for the simulation run
    time_arrival_dict : dict
        Dictionary with (start_time, end_time): arrival_rate_factor
    
    Returns:
    --------
    pd.DataFrame
        Filtered truck dataframe
    """
    truck_data = pd.read_pickle(f'.{run_id}/logs/truck_data.pkl')
    updated_truck_data_path = f'.{run_id}/logs/truck_data.pkl'
    print(truck_data.head())
    filtered_truck_data = filter_data_by_arrival_rate(truck_data, time_arrival_dict, 'arrival')
    filtered_truck_data.to_pickle(updated_truck_data_path)
    return filtered_truck_data

def filter_train_data(run_id, time_arrival_dict):
    """
    Filter train data based on arrival rate factors.
    
    Parameters:
    -----------
    run_id : str
        Unique identifier for the simulation run
    time_arrival_dict : dict
        Dictionary with (start_time, end_time): arrival_rate_factor
    
    Returns:
    --------
    pd.DataFrame
        Filtered train dataframe
    """
    train_data = pd.read_csv(f'.{run_id}/logs/train_data.csv')
    updated_train_data_path = f'.{run_id}/logs/train_data.csv'
    filtered_train_data = filter_data_by_arrival_rate(train_data, time_arrival_dict, 'arrival at')
    filtered_train_data.to_csv(updated_train_data_path, index=False)
    return filtered_train_data

def model_hurricane(env, terminal_resouces, num_terminals_list, terminal_data, run_id, seed):
    """
    This function models a hurricane scenario by reducing the number of berths, cranes, pipelines, and conveyors in the port terminals.
    It simulates the disruption by adjusting the resources available in the terminals between specified time periods.
    Specificcally, the following changes are made:
    - Reduces the number of berths in all container terminals by 1 from time 1165 to 2004.
    - Reduces the number of berths in 10% of liquid and dry bulk terminals by 1 from time 1165 to 2004.
    - Reduces the number of cranes in all container terminals by 1 from time 1165 to 2340.
    - Reduces the number of pipelines in 25% of liquid terminals by 1 from time 1165 to 2340.
    - Reduces the number of conveyors in 25% of dry bulk terminals by 1 from time 1165 to 2340.
    Args:
        env (simpy.Environment): The simulation environment.
        terminal_resouces (tuple): A tuple containing the resources for the port terminals.
        num_terminals_list (list): A list containing the number of container, liquid, and dry bulk terminals.
        terminal_data (dict): A dictionary containing terminal data for crane transfer rates.
        run_id (str): The unique identifier for the simulation run.
        seed (int): The random seed for reproducibility.
    returns:
        None
    """

    random.seed(seed)

    port_berths_container_terminals, port_yard_container_terminals, port_berth_liquid_terminals, port_tanks_liquid_terminals, \
        port_berth_drybulk_terminals, port_silos_drybulk_terminals, port_loading_bays_liquid_terminals, port_drybulk_bays_drybulk_terminals, \
        port_chassis_container_terminals, truck_gates_ctr, truck_gates_liquid, truck_gates_dk, train_loading_racks_ctr, train_loading_racks_liquid, \
        train_loading_racks_dk, day_pilots, night_pilots, tugboats, channel_scheduler = terminal_resouces

    num_container_terminals, num_liquid_terminals, num_drybulk_terminals = num_terminals_list

    if SEVERE_HURRICANE == True and TROPICAL_STORM == False:

        # stop all arrivals between 1500 and 1665
        time_start = HURRICANE_START
        disruption_duration = 24*6

        # No arrivals for 6 days
        stop_and_bulk_arrival(time_start, time_start + disruption_duration, run_id, plot_input=True) # 7 days better

        # recovery periods
        time_end_1_week = time_start + 24*7 # 7 days
        time_end_2_week = time_start + 24*7 + 24*7 # 7 days
        time_end_3_weeks = time_start + 24*7 + 24*14 # 14 days
        time_end_4_weeks = time_start + 24*7 + 24*21 # 21 days

         # For 1st week, 100% of berths have one crane down, 25% of liquid and dry bulk terminals have one pipeline and conveyor down in 50% of berths
        env.process(reduce_cranes(env, -1, list(range(num_container_terminals)),
                    time_start, time_end_1_week, port_berths_container_terminals, terminal_data, 0.3))
        selected_liquid_terminals = random.sample(
            range(num_liquid_terminals), int(0.1 * num_liquid_terminals))
        selected_drybulk_terminals = random.sample(
            range(num_drybulk_terminals), int(0.5 * num_drybulk_terminals)) 
        env.process(reduce_pipelines(env, -1, selected_liquid_terminals,
                    time_start, time_end_1_week, port_berth_liquid_terminals, terminal_data, 0.3))
        env.process(reduce_conveyors(env, -1, selected_drybulk_terminals,
                    time_start, time_end_1_week, port_berth_drybulk_terminals, terminal_data, 0.3))

        # For 1st week, 100% of berths have one crane down, 25% of liquid and dry bulk terminals have one pipeline and conveyor down in 50% of berths
        env.process(reduce_cranes(env, -1, list(range(num_container_terminals)),
                    time_end_1_week, time_end_2_week, port_berths_container_terminals, terminal_data, 0.30))
        env.process(reduce_pipelines(env, -1, selected_liquid_terminals,
                    time_end_1_week, time_end_2_week, port_berth_liquid_terminals, terminal_data, 0.30))
        env.process(reduce_conveyors(env, -1, selected_drybulk_terminals,
                    time_end_1_week, time_end_2_week, port_berth_drybulk_terminals, terminal_data, 0.30))

        # For 2nd week, 25% of berths have one crane down, 25% of (same) liquid and dry bulk terminals have one pipeline and conveyor down in 25% of berths
        env.process(reduce_cranes(env, -1, list(range(num_container_terminals)),
                    time_end_2_week, time_end_3_weeks, port_berths_container_terminals, terminal_data, 0.2))
        env.process(reduce_pipelines(env, -1, selected_liquid_terminals,
                    time_end_2_week, time_end_3_weeks, port_berth_liquid_terminals, terminal_data, 0.2))
        env.process(reduce_conveyors(env, -1, selected_drybulk_terminals,
                    time_end_2_week, time_end_3_weeks, port_berth_drybulk_terminals, terminal_data, 0.2))
        
        # For 3rd week, 10% of berths have one crane down, 25% of (same) liquid and dry bulk terminals have one pipeline and conveyor down in 10% of berths
        env.process(reduce_cranes(env, -1, list(range(num_container_terminals)),
                    time_end_3_weeks, time_end_4_weeks, port_berths_container_terminals, terminal_data, 0.1))
        env.process(reduce_pipelines(env, -1, selected_liquid_terminals,
                    time_end_3_weeks, time_end_4_weeks, port_berth_liquid_terminals, terminal_data, 0.1))
        env.process(reduce_conveyors(env, -1, selected_drybulk_terminals,
                    time_end_3_weeks, time_end_4_weeks, port_berth_drybulk_terminals, terminal_data, 0.1))
        
        # For 4th week, 10% of berths have one crane down, 25% of (same) liquid and dry bulk terminals have one pipeline and conveyor down in 10% of berths
        # env.process(reduce_cranes(env, -1, list(range(num_container_terminals)),
        #             time_end_3_weeks, time_end_1_month, port_berths_container_terminals, terminal_data, 0.1))
        # env.process(reduce_pipelines(env, -1, selected_liquid_terminals,
        #             time_end_3_weeks, time_end_1_month, port_berth_liquid_terminals, terminal_data, 0.1))
        # env.process(reduce_conveyors(env, -1, selected_drybulk_terminals,
        #             time_end_3_weeks, time_end_1_month, port_berth_drybulk_terminals, terminal_data, 0.1))

        # for the first week only 50% of trucks/trains arrive and for the second week only 80% of trucks/trains arrive

        truck_time_arrival_dict = {
            (time_start, time_start + disruption_duration): 0.0, # no trucks during hurricane
            (time_end_1_week, time_end_2_week): 0.1,
            (time_end_2_week, time_end_3_weeks): 0.3,
            (time_end_3_weeks, time_end_4_weeks): 0.5
        }   

        train_time_arrival_dict = {
            (time_start, time_start + disruption_duration): 0.0, # no trains during hurricane
            (time_end_1_week, time_end_2_week): 0.0,
            (time_end_2_week, time_end_3_weeks): 0.5
        }

        filter_truck_data(run_id, truck_time_arrival_dict)
        filter_train_data(run_id, train_time_arrival_dict)

    elif TROPICAL_STORM == True and SEVERE_HURRICANE == False:
        
        time_start = HURRICANE_START
        disruption_duration = 24*4 # 4 days

        time_end_1_week = HURRICANE_START + 24*7 # 7 days
        time_end_2_week = HURRICANE_START + 24*14 # 14 days

        # No damage to port facilities

        # No vessel arrivals for 1 week
        stop_and_bulk_arrival(time_start, time_start + disruption_duration, run_id, plot_input=True)

        # for the first week only 70% of trucks arrive and no change to trains
        truck_time_arrival_dict = {
            (time_start, time_end_1_week): 0.5,
            (time_end_1_week, time_end_2_week): 0.8
        }

        filter_truck_data(run_id, truck_time_arrival_dict)

    elif SEVERE_HURRICANE == True and TROPICAL_STORM == True:
        print("Error: Both SEVERE_HURRICANE and TROPICAL_STORM flags cannot be True at the same time; update the config file.")


def disruption_report_smoothed(smooth_window_points: int = 30, plot_legend: bool = False) -> str:
    """
    Inputs:
        - smooth_window_points: window size (odd integer) for rolling mean smoothing of raw queue
    Outputs (returned as a formatted string):
        1) times: onset, queue_in_point, mobilisation, peak, recovery
        2) a blank line
        3) durations: onset→queue_in_point, queue_in_point→mobilisation, mobilisation→peak, peak→recovery
        4) a blank line
        5) areas: positive (above baseline), negative (below baseline)
    Note: All event times are identified on RAW data, but areas are computed using SMOOTHED data.

    """
    warmup_time = WARMUP_ITERS
    onset_time = HURRICANE_START

    logfileid = f"{int(NUM_MONTHS)}_months_{NUM_RUNS}_runs_run_{LOG_NUM}"
    df_location = f'collatedResults/{logfileid}/data/all_queue.csv'

    df_raw = pd.read_csv(df_location)
    df_raw = df_raw.rename(columns={df_raw.columns[0]: "time"})
    df_raw["mean"] = df_raw.iloc[:, 1:].mean(axis=1)

    time_col = "time"
    queue_col = "mean"
    df = df_raw[[time_col, queue_col]].rename(columns={time_col: "time_hr", queue_col: "q_raw"})

    df = df.sort_values("time_hr").reset_index(drop=True)
    df["time_hr"] = pd.to_numeric(df["time_hr"], errors="coerce")
    df["q_raw"] = pd.to_numeric(df["q_raw"], errors="coerce")
    df = df.dropna(subset=["time_hr", "q_raw"]).reset_index(drop=True)

    # Smoothing for areas only
    w = int(smooth_window_points)
    if w < 3: w = 3
    if w % 2 == 0: w += 1
    df["q_smooth"] = df["q_raw"].rolling(window=w, center=True, min_periods=1).mean()

    # Baseline from [1000, 1500] on RAW
    t0, t1 = warmup_time, onset_time
    base_mask = (df["time_hr"] >= t0) & (df["time_hr"] <= t1)
    baseline = float(df.loc[base_mask, "q_raw"].mean())

    # A: onset
    idx_A = int((df["time_hr"] - onset_time).abs().idxmin())
    A_t = float(df.at[idx_A, "time_hr"]); A_q = float(df.at[idx_A, "q_raw"])

    # t_queue_in_point: point where queue starts to increase (2 positive slopes)
    dQdt = df["q_raw"].diff() / df["time_hr"].diff()
    idx_queue_in = None
    for i in range(idx_A + 1, len(df) - 2):
        if (dQdt.iloc[i] > 0) and (dQdt.iloc[i + 1] > 0):
            idx_queue_in = i; break
    if idx_queue_in is None: idx_queue_in = min(idx_A + 1, len(df) - 1)
    t_queue_in = float(df.at[idx_queue_in, "time_hr"]); q_queue_in = float(df.at[idx_queue_in, "q_raw"])

    # B: mobilisation (first upward crossing to baseline after t_queue_in_point)
    idx_B = None
    for i in range(idx_queue_in, len(df)):
        if df.at[i, "q_raw"] >= baseline:
            idx_B = i; break
    if idx_B is None: idx_B = len(df) - 1
    B_t = float(df.at[idx_B, "time_hr"]); B_q = baseline

    # C: peak (raw max after mobilisation)
    idx_C = int(df.loc[df["time_hr"] >= B_t, "q_raw"].idxmax())
    C_t = float(df.at[idx_C, "time_hr"]); C_q = float(df.at[idx_C, "q_raw"])

    # D: recovery (first downward crossing to baseline after C; linear interpolation on RAW)
    # D_t, D_q = None, None
    # for i in range(idx_C, len(df) - 1):
    #     y0, y1 = df.at[i, "q_raw"], df.at[i + 1, "q_raw"]
    #     x0, x1 = df.at[i, "time_hr"], df.at[i + 1, "time_hr"]
    #     if (y0 > baseline) and (y1 <= baseline):
    #         D_t = float(x0 + (baseline - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)
    #         D_q = baseline
    #         break
    # if D_t is None:
    #     D_t = float(df.at[len(df) - 1, "time_hr"]); D_q = float(df.at[len(df) - 1, "q_raw"])

    # D: recovery (average queue over 2 weeks post-recovery point is <= baseline)
    two_weeks_in_hours = 14 * 24
    D_t, D_q = None, None
    for i in range(idx_C, len(df)):
        # Calculate the end point of the 2-week window
        end_time = df.at[i, "time_hr"] + two_weeks_in_hours

        # Get the subset of data for the 2-week window
        window_mask = (df["time_hr"] >= df.at[i, "time_hr"]) & (df["time_hr"] <= end_time)
        window_data = df.loc[window_mask, "q_raw"]

        # If there's enough data for a full 2-week window, check the average
        if len(window_data) >= 1: # Ensure there is at least one data point
            avg_queue = window_data.mean()
            if avg_queue <= baseline:
                D_t = float(df.at[i, "time_hr"])
                D_q = float(df.at[i, "q_raw"])
                break
    
    # Fallback in case no such point is found
    if D_t is None:
        D_t = float(df.at[len(df) - 1, "time_hr"]); D_q = float(df.at[len(df) - 1, "q_raw"])

    # Durations
    dur_A_queue_in = t_queue_in - A_t
    dur_queue_in_B = B_t - t_queue_in
    dur_B_C = C_t - B_t
    dur_C_D = D_t - C_t

    # Areas between A and D using q_smooth
    mask_AD = (df["time_hr"] >= A_t) & (df["time_hr"] <= D_t)
    xs = df.loc[mask_AD, "time_hr"].to_numpy(dtype=float)
    ys_sm = df.loc[mask_AD, "q_smooth"].to_numpy(dtype=float)
    if xs.size == 0 or xs[-1] < D_t:
        xs = np.append(xs, D_t); ys_sm = np.append(ys_sm, baseline)
    area_pos = float(np.trapz(np.maximum(0.0, ys_sm - baseline), xs))
    area_neg = float(np.trapz(np.maximum(0.0, baseline - ys_sm), xs))

    # Calculate new metrics
    # Robustness (R): R = Q(t_min) / Q_0_bar
    idx_min = int(df.loc[(df["time_hr"] >= A_t) & (df["time_hr"] <= t_queue_in), "q_raw"].idxmin())
    q_min = df.at[idx_min, "q_raw"]
    robustness = q_min / baseline

    # Mobilization (M): M = time from onset to mobilisation
    mobilization = B_t - A_t

    # Peak Queue Increase (P): P = (Q(t_max) - Q_0_bar) / Q_0_bar
    peak_increase = (C_q - baseline) / baseline

    # Max Recovery Rate (m_max): slope of the next 'n' points from the peak
    # Initialize variables to store the most negative slope and the corresponding lookahead
    max_negative_rate = 0.0
    best_lookahead = 0
    best_lookahead_idx = idx_C

    # Loop through lookahead values from 5 to 50
    for recovery_rate_lookahead in range(5, 51):
        idx_C_plus_n = min(idx_C + recovery_rate_lookahead, len(df) - 1)
        if idx_C_plus_n > idx_C:
            rate = (df.at[idx_C_plus_n, "q_raw"] - C_q) / (df.at[idx_C_plus_n, "time_hr"] - C_t)
        else:
            rate = 0.0

        # Check if the current rate is more negative than the current maximum negative rate
        if rate < max_negative_rate:
            max_negative_rate = rate
            best_lookahead = recovery_rate_lookahead
            best_lookahead_idx = idx_C_plus_n



    # drop lat 200 points to avoid edge effects in smoothing
    df = df.iloc[:-200, :].reset_index(drop=True)

    # Plot (PDF): raw & smoothed; shading from smoothed
    logfileid = f"{int(NUM_MONTHS)}_months_{NUM_RUNS}_runs_run_{LOG_NUM}"
    pdf_path = f'collatedResults/{logfileid}/hurricane_report.pdf'
    plt.figure(figsize=(16, 6.5))
    plt.plot(df["time_hr"], df["q_raw"], label="Queue (raw)", alpha=0.5)
    plt.plot(df["time_hr"], df["q_smooth"], label=f"Queue (smoothed)", linewidth=2)
    plt.axhline(baseline, linestyle="--", label="Baseline")
    plt.fill_between(xs, ys_sm, baseline, where=(ys_sm < baseline), alpha=0.3, label="Lost service", color="orange")
    plt.fill_between(xs, ys_sm, baseline, where=(ys_sm > baseline), alpha=0.2, label="Excess queuing", color="pink")

    # shade the warmup period
    plt.axvspan(0, warmup_time, color='gray', alpha=0.2, label="Warmup period")

    # Plot the max recovery rate line segment
    x_rate = [C_t, df.at[best_lookahead_idx, "time_hr"]]
    y_rate = [C_q, df.at[best_lookahead_idx, "q_raw"]]
    plt.plot(x_rate, y_rate, 'g--', label="Max recovery rate", linewidth=2)

    for (xt, yv), lbl in [((A_t, A_q), "Onset"), ((B_t, B_q), "Mobilisation"),
                         ((C_t, C_q), "Peak"), ((D_t, D_q), "Recovery")]:
        plt.scatter([xt], [yv], s=64, label=lbl)
    plt.xlabel("Time (hr)", fontsize=16); plt.ylabel("Queue length", fontsize=16); plt.tight_layout()
    if plot_legend:
        plt.legend(fontsize=14)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.savefig(pdf_path, dpi=200, bbox_inches="tight")
    plt.show()

    lines = [
        f"onset: {A_t:.3f}",
        f"queue-in-point: {t_queue_in:.3f}",
        f"mobilisation: {B_t:.3f}",
        f"peak: {C_t:.3f}",
        f"recovery: {D_t:.3f}",
        "",
        f"time onset to queue-in-point (days): {dur_A_queue_in / 24:.1f}",
        f"time queue-in-point to mobilisation (days): {dur_queue_in_B / 24:.1f}",
        f"time mobilisation to peak (days): {dur_B_C / 24:.1f}",
        f"time peak to recovery (days): {dur_C_D / 24:.1f}",
        "",
        f"area above baseline (vessel days): {area_pos / 24:.2f}",
        f"area below baseline (vessel days): {area_neg / 24:.2f}",
        "",
        f"Robustness: {robustness:.2f}",
        f"Mobilization: {mobilization:.2f}",
        f"Peak Queue Increase: {peak_increase:.2f}",
        f"Max Recovery Rate: {max_negative_rate:.2f}"
    ]

    with open(f'collatedResults/{logfileid}/hurricane_report.txt', 'w') as f:
        for line in lines:
            f.write(line + "\n")
            print(line)

    return "\n".join(lines), str(pdf_path)


def fog_peak_slopes(slope_steps: int = 5, baseline_end: float = 2*7*14):
    """
    - Shades inbound/outbound closure windows.
    - For each pair i, finds the raw peak within the union window [min(in_i, out_i), max(in_i, out_i)].
    - Computes slope from peak -> next `slope_steps` points: (q[t+k]-q[t])/(time[t+k]-time[t]).
    - Produces TWO plots:
        1) Full view: queue with inbound/outbound shaded (no slope segments).
        2) Zoomed to [first_closure_start, last_closure_end]: queue with closures shaded AND slope segments.
    - Prints a compact table (no CSV) with: i, peak_time, peak_queue, slope_peak_to_nextK.

    Uses default Matplotlib colors; markers are circles; no text annotations.
    """
    # ---- Load data ----
    warmup_time = WARMUP_ITERS
    inbound_closed = INBOUND_CLOSED
    outbound_closed = OUTBOUND_CLOSED

    logfileid = f"{int(NUM_MONTHS)}_months_{NUM_RUNS}_runs_run_{LOG_NUM}"
    df_location = f'collatedResults/{logfileid}/data/all_queue.csv'

    df_raw = pd.read_csv(df_location)
    df_raw = df_raw.rename(columns={df_raw.columns[0]: "time"})
    df_raw["mean"] = df_raw.iloc[:, 1:].mean(axis=1)


    time_col = "time"
    mean_col = "mean"
    df = df_raw[[time_col, mean_col]].rename(columns={time_col:"time_hr", mean_col:"q_raw"})

    df = df.sort_values("time_hr").reset_index(drop=True)
    df["time_hr"] = pd.to_numeric(df["time_hr"], errors="coerce")
    df["q_raw"]   = pd.to_numeric(df["q_raw"], errors="coerce")
    df = df.dropna(subset=["time_hr","q_raw"]).reset_index(drop=True)

    # Baseline from 1000 to 1500 (raw)
    base_mask = (df["time_hr"] >= warmup_time) & (df["time_hr"] <= baseline_end)
    baseline = float(df.loc[base_mask, "q_raw"].mean())

    # ---- Per-pair peak and slope to next K steps ----
    results = []
    n = min(len(inbound_closed), len(outbound_closed))
    for i in range(n):
        a0, a1 = inbound_closed[i]
        b0, b1 = outbound_closed[i]
        win_start = min(a0, b0)
        win_end   = max(a1, b1)
        mask_win = (df["time_hr"] >= win_start) & (df["time_hr"] <= win_end)
        if mask_win.sum() == 0:
            continue
        idx_peak = int(df.loc[mask_win, "q_raw"].idxmax())
        t_peak = float(df.at[idx_peak, "time_hr"])
        q_peak = float(df.at[idx_peak, "q_raw"])
        
        # Check if the queue decreases after the peak
        idx_k = min(idx_peak + max(1, int(slope_steps)), len(df)-1)
        t_k = float(df.at[idx_k, "time_hr"])
        q_k = float(df.at[idx_k, "q_raw"])
        
        # Only calculate and append the result if the queue decreases
        if q_k <= q_peak:
            slope = (q_k - q_peak)/(t_k - t_peak) if (t_k - t_peak) != 0 else np.nan
            results.append({
                "i": i+1,
                "peak_time": t_peak,
                "peak_queue": q_peak,
                "nextK_time": t_k,
                "nextK_queue": q_k,
                "slope_peak_to_nextK": slope
            })

    # ---- PLOTS ----
    # 1) Full view: open/closed only
    pdf_full = f'collatedResults/{logfileid}/fog_report_full.pdf'
    export_prefix = f'collatedResults/{logfileid}/fog_report'
    
    plt.figure(figsize=(14,7))
    plt.plot(df["time_hr"], df["q_raw"], label="Queue")
    plt.axhline(baseline, linestyle="--", label="Baseline")
    for (s,e) in inbound_closed:
        plt.axvspan(s, e, alpha=0.18, color= 'green', label="Inbound closed" if (s,e)==inbound_closed[0] else None)
    for (s,e) in outbound_closed:
        plt.axvspan(s, e, alpha=0.18, color = 'orange', label="Outbound closed" if (s,e)==outbound_closed[0] else None)
    plt.xlabel("Time (hr)", fontsize=16); plt.ylabel("Queue length", fontsize=16); plt.legend(loc="best", fontsize=16)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.tight_layout(); plt.savefig(pdf_full, dpi=200, bbox_inches="tight"); plt.show(); plt.close()

    # 2) Zoomed: show slopes too
    pdf_zoom = Path(f"{export_prefix}_zoom.pdf")
    fog_start = min(min(x[0] for x in inbound_closed), min(x[0] for x in outbound_closed))
    fog_end   = max(max(x[1] for x in inbound_closed), max(x[1] for x in outbound_closed))
    m = (df["time_hr"] >= fog_start) & (df["time_hr"] <= fog_end)
    plt.figure(figsize=(14,7))
    plt.plot(df.loc[m, "time_hr"], df.loc[m, "q_raw"], label="Queue")
    plt.axhline(baseline, linestyle="--", label="Baseline")
    for (s,e) in inbound_closed:
        if e>=fog_start and s<=fog_end:
            plt.axvspan(s, e, alpha=0.18, color = 'green', label="Inbound closed" if (s,e)==inbound_closed[0] else None)
    for (s,e) in outbound_closed:
        if e>=fog_start and s<=fog_end:
            plt.axvspan(s, e, alpha=0.18, color = 'orange', label="Outbound closed" if (s,e)==outbound_closed[0] else None)
    # circle markers + slope segments
    for r in results:
        if fog_start <= r["peak_time"] <= fog_end:
            plt.plot([r["peak_time"]], [r["peak_queue"]], marker='o', linestyle='None', color='green', markersize=8)
            plt.plot([r["peak_time"], r["nextK_time"]],
                     [r["peak_queue"], r["nextK_queue"]],
                     linestyle='--', linewidth=3, color = 'k')
    plt.xlabel("Time (hr)",fontsize=16); plt.ylabel("Queue length",fontsize=16)
    plt.legend(
        loc="upper center",           # anchor relative to top center
        bbox_to_anchor=(0.5, -0.1),  # shift it below the axes
        fontsize=16,
        ncol=4                        # optional: spread entries into multiple columns
    )
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.tight_layout(); plt.savefig(pdf_zoom, dpi=200, bbox_inches="tight"); plt.show(); plt.close()

    # ---- PRINT TABLE ----
    print("i, peak_time, peak_queue, slope_peak_to_nextK")
    for r in results:
        print(f"{r['i']}, {r['peak_time']:.3f}, {r['peak_queue']:.3f}, {r['slope_peak_to_nextK']:.6f}")
    print("\nAverage slope is", np.mean([r["slope_peak_to_nextK"] for r in results]))

    # save results as a text file
    with open(f'collatedResults/{logfileid}/fog_report.txt', 'w') as f:
        f.write("i, peak_time, peak_queue, slope_peak_to_nextK\n")
        for r in results:
            f.write(f"{r['i']}, {r['peak_time']:.3f}, {r['peak_queue']:.3f}, {r['slope_peak_to_nextK']:.6f}\n")
            
    return str(pdf_full), str(pdf_zoom)



              