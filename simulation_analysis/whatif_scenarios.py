"""
This module contains functions to model disruptions in a port simulation environment.
"""
import random

import pandas as pd
import matplotlib.pyplot as plt

from simulation_classes.port import Crane, Pipeline, Conveyor
from simulation_handler.helpers import get_value_by_terminal



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

    # stop all arrivals between 1500 and 1665
    stop_and_bulk_arrival(1500, 1665, run_id, plot_input=True)

    # TODO: Reduce truck train access

    # reduce berths in all ctr terminals by 1 from 1165 to 2004
    env.process(reduce_berths(
        env, -1, [0, 1], 1165, 2004, port_berths_container_terminals))

    # # select 25% of liquid and drybulk terminals and reduce berths by 1 from 1165 to 2004
    selected_liquid_terminals = random.sample(
        range(num_liquid_terminals), int(0.1*num_liquid_terminals))
    selected_drybulk_terminals = random.sample(
        range(num_drybulk_terminals), int(0.1*num_drybulk_terminals))
    env.process(reduce_berths(env, -1, selected_liquid_terminals,
                1165, 2004, port_berth_liquid_terminals))
    env.process(reduce_berths(env, -1, selected_drybulk_terminals,
                1165, 2004, port_berth_drybulk_terminals))

    # reduce cranes, pipelimnes and conveyors in all ctr terminals and 25% of liq and dbk terminals by 1 from 1165 to 2340 by 1 from 25% of berth
    selected_liquid_terminals = random.sample(
        range(num_liquid_terminals), int(0.25*num_liquid_terminals))
    selected_drybulk_terminals = random.sample(
        range(num_drybulk_terminals), int(0.25*num_drybulk_terminals))

    env.process(reduce_cranes(env, -1, list(range(num_container_terminals)),
                1165, 2340, port_berths_container_terminals, terminal_data, 0.25))
    env.process(reduce_pipelines(env, -1, selected_liquid_terminals,
                1165, 2340, port_berth_liquid_terminals, terminal_data, 0.25))
    env.process(reduce_conveyors(env, -1, selected_drybulk_terminals,
                1165, 2340, port_berth_drybulk_terminals, terminal_data, 0.25))
