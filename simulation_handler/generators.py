"""
Generates resources and processes needed to run the simulation using the object classes defined in truck.py, Channel.py, and Port.py 
and the DES processes contained in ContainerTerminal.py, LiquidTerminal.py, and DryBulkTerminal.py.
"""

import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import constants
from simulation_classes.terminal_container import ContainerTerminal
from simulation_classes.terminal_liquid import LiquidTerminal
from simulation_classes.terminal_drybulk import DryBulkTerminal
from simulation_classes.truck import Truck
from simulation_classes.train import Train
from simulation_classes.pipeline import Pipeline
from simulation_classes.port import Container, get_value_by_terminal, get_value_from_terminal_tuple
from simulation_classes.channel import Channel
from simulation_analysis.results import plot_queues, track_utilization, save_track_list, get_utilization

SIMULATION_TIME = constants.SIMULATION_TIME
PIPELINE_RATE = constants.PIPELINE_RATE
CHANNEL_SAFETWOWAY = constants.CHANNEL_SAFETWOWAY


def initialize_rng(seed):
    """
    Initialize random state objects for repeatable randomness across runs.
    This function sets the global random number generator states for both the built-in `random` module and NumPy's random module.
    This is useful for ensuring that the simulation can be reproduced with the same random events.
    This function should be called at the start of the simulation to ensure that all random processes are initialized with the same seed.
    Args:
        seed (int): The seed value for the random number generator.
    Returns:    
        None
    """
    global rng_random, rng_numpy
    rng_random = random.Random(seed)
    rng_numpy = np.random.default_rng(seed)


def update_availability(run_id, env, it, port_berths_container_terminals, port_yard_container_terminals, port_berths_liquid_terminals, port_tanks_liquid_terminals, port_berths_drybulk_terminals, port_silos_drybulk_terminals, availability_df_container, availability_df_liquid, availability_df_drybulk, ship_data, terminal_data):
    """
    This function updates the availability of berths, yards, tanks, and silos at the port terminals.
    It calculates the number of available and used resources for each terminal type (Container, Liquid, DryBulk) and appends this data to the respective DataFrames.
    This function also tracks the queue lengths at each terminal type and updates the availability DataFrames with the current time and resource availability.

    Args:
        run_id (str): The unique identifier for the simulation run.
        env (simpy.Environment): The simulation environment.
        it (int): The current iteration of the simulation.
        port_berths_container_terminals (list): List of container terminal berths.
        port_yard_container_terminals (list): List of container terminal yards.
        port_berths_liquid_terminals (list): List of liquid terminal berths.
        port_tanks_liquid_terminals (list): List of liquid terminal tanks.
        port_berths_drybulk_terminals (list): List of dry bulk terminal berths.
        port_silos_drybulk_terminals (list): List of dry bulk terminal silos.
        availability_df_container (pd.DataFrame): DataFrame to store container terminal availability data.
        availability_df_liquid (pd.DataFrame): DataFrame to store liquid terminal availability data.
        availability_df_drybulk (pd.DataFrame): DataFrame to store dry bulk terminal availability data.
        ship_data (dict): Dictionary containing ship data for the simulation.
        terminal_data (dict): Dictionary containing terminal data for the simulation.
    Returns:
        availability_df_container (pd.DataFrame): Updated DataFrame with container terminal availability data.
        availability_df_liquid (pd.DataFrame): Updated DataFrame with liquid terminal availability data.
        availability_df_drybulk (pd.DataFrame): Updated DataFrame with dry bulk terminal availability data.
        queue_lengths_in_ctr (list): List of queue lengths in container terminals.
        queue_lengths_in_liq (list): List of queue lengths in liquid terminals.
        queue_lengths_in_drybulk (list): List of queue lengths in dry bulk terminals.
    """
    # Initialize lists to store vessel availability data for each terminal
    berth_availability_data_container = []
    yard_availability_data = []
    berth_availability_data_liquid = []
    tank_availability_data = []
    berth_availability_data_drybulk = []
    silo_availability_data = []
    queue_lengths_in_ctr = []
    queue_lengths_in_liq = []
    queue_lengths_in_drybulk = []

    # Container Terminals
    for i, (port_berths, port_yard) in enumerate(zip(port_berths_container_terminals, port_yard_container_terminals)):
        # Determine the vessel queue length and number of available berths for a container terminal
        queue = len(port_berths.get_queue)
        queue_lengths_in_ctr.append(queue)
        available_port_berths = len(port_berths.items)
        used_port_berths = get_value_by_terminal(
            terminal_data, "Container", i+1, "Berths") - available_port_berths
        berth_availability_data_container.extend(
            [available_port_berths, used_port_berths, queue])

        # Determine the number of containers stored on the dock (yard) and the remaining available storage space
        used_yard = len(port_yard.items)
        available_yard = get_value_by_terminal(
            terminal_data, "Container", i+1, "storage volume") - used_yard
        yard_availability_data.extend(
            [available_yard, used_yard])

    # Liquid Terminals
    for i, port_berths in enumerate(port_berths_liquid_terminals):
        # Determine the vessel queue length and number of available berths for a liquid bulk terminal
        queue = len(port_berths.get_queue)
        queue_lengths_in_liq.append(queue)
        available_port_berths = len(port_berths.items)
        used_port_berths = get_value_by_terminal(
            terminal_data, "Liquid", i+1, "Berths") - available_port_berths
        berth_availability_data_liquid.extend(
            [available_port_berths, used_port_berths, queue])

    for i, port_tanks in enumerate(port_tanks_liquid_terminals):
        # Determine the amount of storage space used and available in the liquid bulk tanks
        used_tank = port_tanks.level
        available_tank = get_value_by_terminal(
            terminal_data, "Liquid", i+1, "storage volume") - used_tank
        tank_availability_data.extend([available_tank, used_tank])

    # Dry Bulk Terminals
    for i, port_berths in enumerate(port_berths_drybulk_terminals):
        # Determine the vessel queue length and number of available berths for a dry bulk terminal
        queue = len(port_berths.get_queue)
        queue_lengths_in_drybulk.append(queue)
        available_port_berths = len(port_berths.items)
        used_port_berths = get_value_by_terminal(
            terminal_data, "DryBulk", i+1, "Berths") - available_port_berths
        berth_availability_data_drybulk.extend(
            [available_port_berths, used_port_berths, queue])

    for i, port_silos in enumerate(port_silos_drybulk_terminals):
        # Determine the amount of storage space used and available in the dry bulk silos
        used_silo = port_silos.level
        available_silo = get_value_by_terminal(
            terminal_data, "DryBulk", i+1, "storage volume") - used_silo
        silo_availability_data.extend([available_silo, used_silo])

    # Update DataFrames for each terminal type to track storage availability over the span of the simulation
    availability_container = [
        env.now] + berth_availability_data_container + yard_availability_data
    columns_container = ["Time"] + [f"Terminal_{i+1}_{col}" for i in range(len(port_berths_container_terminals)) for col in ["Available_Berth_Ctr", "Used_Berth_Ctr", "Berth_Queue_Ctr"]] + [
        f"Terminal_{i+1}_{col}" for i in range(len(port_yard_container_terminals)) for col in ["Available_Yard", "Used_Yard"]]
    row_container = pd.DataFrame(
        [availability_container], columns=columns_container)
    availability_df_container = pd.concat(
        [availability_df_container, row_container], ignore_index=True)

    availability_liquid = [env.now] + \
        berth_availability_data_liquid + tank_availability_data
    columns_liquid = ["Time"] + [f"Terminal_{i+1}_{col}" for i in range(len(port_berths_liquid_terminals)) for col in ["Available_Berth_liq", "Used_Berth_liq", "Berth_Queue_liq"]] + [
        f"Terminal_{i+1}_{col}" for i in range(len(port_tanks_liquid_terminals)) for col in ["Available_Tank", "Used_Tank"]]
    row_liquid = pd.DataFrame([availability_liquid], columns=columns_liquid)
    availability_df_liquid = pd.concat(
        [availability_df_liquid, row_liquid], ignore_index=True)

    availability_drybulk = [env.now] + \
        berth_availability_data_drybulk + silo_availability_data
    columns_drybulk = ["Time"] + [f"Terminal_{i+1}_{col}" for i in range(len(port_berths_drybulk_terminals)) for col in ["Available_Berth_db", "Used_Berth_db", "Berth_Queue_db"]] + [
        f"Terminal_{i+1}_{col}" for i in range(len(port_silos_drybulk_terminals)) for col in ["Available_Silo", "Used_Silo"]]
    row_drybulk = pd.DataFrame([availability_drybulk], columns=columns_drybulk)
    availability_df_drybulk = pd.concat(
        [availability_df_drybulk, row_drybulk], ignore_index=True)

    track_list = get_utilization(
        availability_df_container, availability_df_liquid, availability_df_drybulk, run_id)

    # innitialise empty dataframes
    if it == 0:
        availability_df_liquid = pd.DataFrame()
        availability_df_drybulk = pd.DataFrame()
    if it > 2:  # save from second iterarion
        save_track_list(run_id, env.now, track_list)
    if it == len(ship_data):
        # Write the data to an Excel file, with different sheets for Container, Liquid Bulk, and Dry Bulk Terminals.
        file_exists = os.path.isfile("./logs/availability.xlsx")
        if file_exists:
            mode = 'a'
            if_sheet_exists = 'replace'
        else:
            mode = 'w'
            if_sheet_exists = None  # Not used in write mode

        with pd.ExcelWriter(f".{run_id}/logs/availability.xlsx", engine="openpyxl", mode=mode, if_sheet_exists=if_sheet_exists) as writer:
            availability_df_container.to_excel(
                writer, sheet_name="Container_Terminals", index=False)
            availability_df_liquid.to_excel(
                writer, sheet_name="Liquid_Terminals", index=False)
            availability_df_drybulk.to_excel(
                writer, sheet_name="Dry_Bulk_Terminals", index=False)

    return availability_df_container, availability_df_liquid, availability_df_drybulk, queue_lengths_in_ctr, queue_lengths_in_liq, queue_lengths_in_drybulk


def create_containers(ship_id, ship_info):
    """
    This function creates a list of Container objects to be loaded and unloaded from a ship.
    It generates the number of containers based on the ship's information, specifically the width and the number of containers to load and unload.
    Args:
        ship_id (str): The unique identifier for the ship.
        ship_info (dict): A dictionary containing information about the ship, including its width and the number of containers to load and unload.
    Returns:
        containers_to_unload (list): A list of Container objects representing the containers to be unloaded from the ship.
        containers_to_load (list): A list of Container objects representing the containers to be loaded onto the ship.
    """
    containers_to_unload = [Container(id=f"{ship_id}_unload_{j}", width=ship_info['width']) for j in range(
        int(ship_info['num_container_or_liq_tons_or_dry_tons_to_unload']))]
    containers_to_load = [Container(id=f"{ship_id}_load_{j}", width=ship_info['width'])for j in range(
        int(ship_info['num_container_or_liq_tons_or_dry_tons_to_load']))]
    return containers_to_unload, containers_to_load


def tons_to_unload_load(ship_id, ship_info):
    """
    This function retrieves the number of tons to unload and load for a ship based on its information.
    Args:
        ship_id (str): The unique identifier for the ship.
        ship_info (dict): A dictionary containing information about the ship, including the number of tons to unload and load.
    Returns:
        unload_tons (int): The number of tons to unload from the ship.
        load_tons (int): The number of tons to load onto the ship.
    """
    unload_tons = ship_info['num_container_or_liq_tons_or_dry_tons_to_unload']
    load_tons = ship_info['num_container_or_liq_tons_or_dry_tons_to_load']
    return unload_tons, load_tons



def ship_generator(run_id, env, chassis_bays_utilization, port_berths_container_terminals, port_yard_container_terminals, port_berths_liquid_terminals, port_tanks_liquid_terminals,
                   port_berths_drybulk_terminals, port_silos_drybulk_terminals, channel, day_pilots, night_pilots, tugboats, events, ship_logs, channel_events, channel_logs, SHIPS_IN_ANCHORAGE, SHIPS_IN_CHANNEL, ship_data, terminal_data, liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink):
    """
    This function generates ships and processes them in the simulation environment.
    It iterates through the ship data, simulating the arrival of each ship at the port and processing it based on its type (Container, Liquid, DryBulk).
    The ContainerTerminal, LiquidTerminal, and DryBulkTerminal classes simulates the vessel's operations at the port.
    It updates the availability of resources at the port terminals and tracks the number of ships in the channel and anchorage.
    Args:
        run_id (str): The unique identifier for the simulation run.
        env (simpy.Environment): The simulation environment.
        chassis_bays_utilization (dict): A dictionary to track chassis bays utilization.
        port_berths_container_terminals (list): List of container terminal berths.
        port_yard_container_terminals (list): List of container terminal yards.
        port_berths_liquid_terminals (list): List of liquid terminal berths.
        port_tanks_liquid_terminals (list): List of liquid terminal tanks.
        port_berths_drybulk_terminals (list): List of dry bulk terminal berths.
        port_silos_drybulk_terminals (list): List of dry bulk terminal silos.
        channel (Channel): The channel object for managing ship movements.
        day_pilots (int): Number of day pilots available.
        night_pilots (int): Number of night pilots available.
        tugboats (int): Number of tugboats available.
        events (list): List to store events during the simulation.
        ship_logs (list): List to store ship logs during the simulation.
        channel_events (list): List to store channel events during the simulation.
        channel_logs (list): List to store channel logs during the simulation.
        SHIPS_IN_ANCHORAGE (list): List to track ships in anchorage by type.
        SHIPS_IN_CHANNEL (list): List to track ships in the channel.
        ship_data (dict): Dictionary containing ship data for the simulation.
        terminal_data (dict): Dictionary containing terminal data for the simulation.
        liq_terminals_with_pipeline_source (list): List of liquid terminals with pipeline source connections.
        liq_terminals_with_pipeline_sink (list): List of liquid terminals with pipeline sink connections.
    Yields:
        simpy.Timeout: A timeout event to simulate the arrival of each ship at the port.
    Returns:
        None
    """ 
    i = 0
    terminal_data = terminal_data

    availability_df_container = pd.DataFrame()
    availability_df_liquid = pd.DataFrame()
    availability_df_drybulk = pd.DataFrame()

    queue_ctr_its = []
    queue_liq_its = []
    queue_drybulk_its = []

    SHIPS_IN_CHANNEL_TRACK = []
    SHIPS_IN_ANCHORAGE_TRACK = []
    CONTAINER_VES_IN_ANCHORAGE = []
    LIQUID_VES_IN_ANCHORAGE = []
    DRYBULK_VES_IN_ANCHORAGE = []

    it = 0

    for ship_id, ship_info in tqdm(ship_data.items(), desc=f"Simulation Progress {run_id}"):
        it += 1
        yield env.timeout(ship_info['arrival'] - env.now)
        if ship_data[ship_id]['ship_type'] == 'Container':
            SHIPS_IN_ANCHORAGE[0] += 1
        elif ship_data[ship_id]['ship_type'] == 'Liquid':
            SHIPS_IN_ANCHORAGE[1] += 1
        elif ship_data[ship_id]['ship_type'] == 'DryBulk':
            SHIPS_IN_ANCHORAGE[2] += 1

        selected_port = ship_info["terminal"] - 1
        last_section = ship_info["last_section"]

        # Container ship processes
        if ship_info['ship_type'] == 'Container':
            port_berths = port_berths_container_terminals[selected_port]
            port_yard = port_yard_container_terminals[selected_port]
            selected_terminal = selected_port + 1
            availability_df_container, availability_df_liquid, availability_df_drybulk, queue_lengths_in_ctr, queue_lengths_in_liq, queue_lengths_in_drybulk = update_availability(run_id, env, it, port_berths_container_terminals,
                                                                                                                                                                                   port_yard_container_terminals, port_berths_liquid_terminals, port_tanks_liquid_terminals, port_berths_drybulk_terminals, port_silos_drybulk_terminals, availability_df_container, availability_df_liquid, availability_df_drybulk, ship_data, terminal_data)
            queue_ctr_its.append(queue_lengths_in_ctr)
            containers_to_unload, containers_to_load = create_containers(
                ship_id, ship_info)
            transfer_rate = get_value_by_terminal(
                terminal_data, "Container", terminal_id=ship_info["terminal"], resource_name="transfer rate per unit")
            transfer_time = 1 / transfer_rate
            ContainerTerminal(env, chassis_bays_utilization, run_id, channel, day_pilots, night_pilots, tugboats, ship_info, last_section, selected_terminal=selected_terminal, id=ship_id, ship_type=ship_info['ship_type'], draft=ship_info['draft'],
                              width=ship_info['width'], unload_time_per_container=transfer_time, load_time_per_container=transfer_time, containers_to_unload=containers_to_unload, containers_to_load=containers_to_load, events=events, ship_logs=ship_logs, port_berths=port_berths, port_yard=port_yard, SHIPS_IN_CHANNEL=SHIPS_IN_CHANNEL, SHIPS_IN_ANCHORAGE=SHIPS_IN_ANCHORAGE, terminal_data=terminal_data)
       
        # Liquid and Dry Bulk ship processes
        elif ship_info['ship_type'] == 'Liquid':
            port_berths = port_berths_liquid_terminals[selected_port]
            port_tanks = port_tanks_liquid_terminals[selected_port]
            selected_terminal = selected_port + 1
            availability_df_container, availability_df_liquid, availability_df_drybulk, queue_lengths_in_ctr, queue_lengths_in_liq, queue_lengths_in_drybulk = update_availability(run_id, env, it, port_berths_container_terminals,
                                                                                                                                                                                   port_yard_container_terminals, port_berths_liquid_terminals, port_tanks_liquid_terminals, port_berths_drybulk_terminals, port_silos_drybulk_terminals, availability_df_container, availability_df_liquid, availability_df_drybulk, ship_data, terminal_data)
            queue_liq_its.append(queue_lengths_in_liq)
            unload_tons, load_tons = tons_to_unload_load(ship_id, ship_info)
            transfer_rate = get_value_by_terminal(
                terminal_data, "Liquid", terminal_id=ship_info["terminal"], resource_name="transfer rate per unit")
            transfer_time = 1 / transfer_rate
            LiquidTerminal(env, chassis_bays_utilization, run_id, channel, day_pilots, night_pilots, tugboats, ship_info, last_section, selected_terminal, id=ship_id, ship_type=ship_info['ship_type'], draft=ship_info['draft'],
                           width=ship_info['width'], unload_time=transfer_time, load_time=transfer_time, unload_tons=unload_tons, load_tons=load_tons, events=events, ship_logs=ship_logs, port_berths=port_berths, port_tanks=port_tanks, SHIPS_IN_CHANNEL=SHIPS_IN_CHANNEL, SHIPS_IN_ANCHORAGE=SHIPS_IN_ANCHORAGE, terminal_data=terminal_data,  liq_terminals_with_pipeline_source=liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink=liq_terminals_with_pipeline_sink)
        
        # Dry Bulk ship processes
        elif ship_info['ship_type'] == 'DryBulk':
            port_berths = port_berths_drybulk_terminals[selected_port]
            port_silos = port_silos_drybulk_terminals[selected_port]
            selected_terminal = selected_port + 1
            availability_df_container, availability_df_liquid, availability_df_drybulk, queue_lengths_in_ctr, queue_lengths_in_liq, queue_lengths_in_drybulk = update_availability(run_id, env, it, port_berths_container_terminals,
                                                                                                                                                                                   port_yard_container_terminals, port_berths_liquid_terminals, port_tanks_liquid_terminals, port_berths_drybulk_terminals, port_silos_drybulk_terminals, availability_df_container, availability_df_liquid, availability_df_drybulk, ship_data, terminal_data)
            queue_drybulk_its.append(queue_lengths_in_drybulk)
            unload_tons, load_tons = tons_to_unload_load(ship_id, ship_info)
            transfer_rate = get_value_by_terminal(
                terminal_data, "DryBulk", terminal_id=ship_info["terminal"], resource_name="transfer rate per unit")
            transfer_time = 1 / transfer_rate
            DryBulkTerminal(env, chassis_bays_utilization, run_id, channel, day_pilots, night_pilots, tugboats, ship_info, last_section, selected_terminal=selected_terminal, id=ship_id, ship_type=ship_info['ship_type'], draft=ship_info
                            ['draft'], width=ship_info['width'], unload_time=transfer_time, load_time=transfer_time, unload_tons=unload_tons, load_tons=load_tons, events=events, ship_logs=ship_logs, port_berths=port_berths, port_silos=port_silos, SHIPS_IN_CHANNEL=SHIPS_IN_CHANNEL, SHIPS_IN_ANCHORAGE=SHIPS_IN_ANCHORAGE, terminal_data=terminal_data)

        i += 1      # Move to the next vessel in the input file
        SHIPS_IN_CHANNEL_TRACK.append((env.now, SHIPS_IN_CHANNEL[0]))
        SHIPS_IN_ANCHORAGE_TRACK.append(
            (env.now, SHIPS_IN_ANCHORAGE[0] + SHIPS_IN_ANCHORAGE[1] + SHIPS_IN_ANCHORAGE[2]))
        CONTAINER_VES_IN_ANCHORAGE.append((env.now, SHIPS_IN_ANCHORAGE[0]))
        LIQUID_VES_IN_ANCHORAGE.append((env.now, SHIPS_IN_ANCHORAGE[1]))
        DRYBULK_VES_IN_ANCHORAGE.append((env.now, SHIPS_IN_ANCHORAGE[2]))



def truck_generator(run_id, liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink, chassis_bays_utilization, env, terminal_tuple_cache, port_tanks_liquid_terminals, port_yard_container_terminals, port_silos_drybulk_terminals, port_loading_bays_liquid_terminals, port_drybulk_bays_drybulk_terminals, port_chassis_container_terminals, truck_gates_ctr, truck_gates_liquid, truck_gates_dk, events, seed, terminal_data):
    """
    This function generates trucks and processes them in the simulation environment.
    It iterates through the truck data, simulating the arrival of each truck at the port and processing it based on its type (Container, Liquid, DryBulk).
    It creates Truck objects and initializes them with the appropriate parameters based on the truck type and terminal information. (This simulates each truck's processing at the port.)
    Args:
        run_id (str): The unique identifier for the simulation run.
        liq_terminals_with_pipeline_source (list): List of liquid terminals with pipeline source connections.
        liq_terminals_with_pipeline_sink (list): List of liquid terminals with pipeline sink connections.   
        chassis_bays_utilization (dict): A dictionary to track chassis bays utilization.
        env (simpy.Environment): The simulation environment.
        terminal_tuple_cache (dict): A cache of terminal tuples for quick access to terminal data.
        port_tanks_liquid_terminals (list): List of liquid terminal tanks.
        port_yard_container_terminals (list): List of container terminal yards.
        port_silos_drybulk_terminals (list): List of dry bulk terminal silos.
        port_loading_bays_liquid_terminals (list): List of liquid terminal loading bays.
        port_drybulk_bays_drybulk_terminals (list): List of dry bulk terminal loading bays.
        port_chassis_container_terminals (list): List of container terminal chassis bays.
        truck_gates_ctr (list): List of container terminal truck gates.
        truck_gates_liquid (list): List of liquid terminal truck gates.
        truck_gates_dk (list): List of dry bulk terminal truck gates.
        events (list): List to store events during the simulation.
        seed (int): The seed value for the random number generator.
        terminal_data (dict): Dictionary containing terminal data for the simulation.
    Yields:
        simpy.Timeout: A timeout event to simulate the arrival of each truck at the port.
    Returns:
        None
    """

    initialize_rng(seed)

    truck_data = pd.read_pickle(f".{run_id}/logs/truck_data.pkl")
    truck_data = truck_data.to_dict(orient="index")

    for truck_id, truck_info in truck_data.items():
        yield env.timeout(truck_info['arrival'] - env.now)
        if truck_info['truck_type'] == 'Container':
            terminal_type = "Container"
            terminal_id = truck_info['terminal_id']
            port_tanks = None
            port_yard = port_yard_container_terminals[terminal_id]
            port_silos = None
            loading_bays = None
            drybulk_bays = None
            truck_chassis = port_chassis_container_terminals[terminal_id]
            container_load_amount = get_value_from_terminal_tuple(
                terminal_tuple_cache, "Container", terminal_id=terminal_id+1, resource_name="truck payload size")
            container_unload_amount = get_value_from_terminal_tuple(
                terminal_tuple_cache, "Container", terminal_id=terminal_id+1, resource_name="truck payload size")
            container_amount = (container_load_amount, container_unload_amount)
            Truck(env, liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink, chassis_bays_utilization, truck_id, run_id, terminal_type, terminal_id+1, container_amount,
                  None, None, loading_bays, port_tanks, truck_chassis, port_yard, port_silos, drybulk_bays, events, seed=rng_random.randint(1, 100000000), terminal_data=terminal_data)
        elif truck_info['truck_type'] == 'Liquid':
            terminal_type = "Liquid"
            terminal_id = truck_info['terminal_id']
            port_tanks = port_tanks_liquid_terminals[terminal_id]
            port_yard = None
            port_silos = None
            loading_bays = port_loading_bays_liquid_terminals[terminal_id]
            drybulk_bays = None
            truck_chassis = None
            liquid_amount = get_value_from_terminal_tuple(
                terminal_tuple_cache, "Liquid", terminal_id=terminal_id+1, resource_name="truck payload size")
            Truck(env, liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink, chassis_bays_utilization, truck_id, run_id, terminal_type, terminal_id+1, None, liquid_amount,
                  None, loading_bays, port_tanks, truck_chassis, port_yard, port_silos, drybulk_bays, events, seed=rng_random.randint(1, 100000000), terminal_data=terminal_data)
        elif truck_info['truck_type'] == 'DryBulk':
            terminal_type = "DryBulk"
            terminal_id = truck_info['terminal_id']
            port_tanks = None
            port_yard = None
            port_silos = port_silos_drybulk_terminals[terminal_id]
            loading_bays = None
            drybulk_bays = port_drybulk_bays_drybulk_terminals[terminal_id]
            truck_chassis = None
            drybulk_amount = get_value_from_terminal_tuple(
                terminal_tuple_cache, "DryBulk", terminal_id=terminal_id+1, resource_name="truck payload size")
            Truck(env, liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink, chassis_bays_utilization, truck_id, run_id, terminal_type, terminal_id+1, None, None,
                  drybulk_amount, loading_bays, port_tanks, truck_chassis, port_yard, port_silos, drybulk_bays, events, seed=rng_random.randint(1, 100000000), terminal_data=terminal_data)


def train_generator(run_id, liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink, env, terminal_tuple_cache, train_loading_racks_ctr, train_loading_racks_liquid, train_loading_racks_dk, train_events, port_tanks_liquid_terminals, port_yard_container_terminals, port_silos_drybulk_terminals, seed):
    """
    This function generates trains and processes them in the simulation environment.
    It iterates through the train data, simulating the arrival of each train at the port and processing it based on its cargo type (Container, Liquid, DryBulk).
    It creates Train objects and initializes them with the appropriate parameters based on the train type and terminal information. (This simulates each train's processing at the port.)
    Args:
        run_id (str): The unique identifier for the simulation run.
        liq_terminals_with_pipeline_source (list): List of liquid terminals with pipeline source connections.
        liq_terminals_with_pipeline_sink (list): List of liquid terminals with pipeline sink connections.
        env (simpy.Environment): The simulation environment.
        terminal_tuple_cache (dict): A cache of terminal tuples for quick access to terminal data.
        train_loading_racks_ctr (dict): Dictionary of loading racks for container trains.
        train_loading_racks_liquid (dict): Dictionary of loading racks for liquid trains.
        train_loading_racks_dk (dict): Dictionary of loading racks for dry bulk trains.
        train_events (list): List to store train events during the simulation.
        port_tanks_liquid_terminals (list): List of liquid terminal tanks.
        port_yard_container_terminals (list): List of container terminal yards.
        port_silos_drybulk_terminals (list): List of dry bulk terminal silos.
        seed (int): The seed value for the random number generator.
    Yields:
        simpy.Timeout: A timeout event to simulate the arrival of each train at the port.
    Returns:
        None
    """
    initialize_rng(seed)

    train_df = pd.read_csv(f".{run_id}/logs/train_data.csv")
    train_data = train_df.to_dict(orient="index")

    for train_id, train_info in train_data.items():
        yield env.timeout(train_info['arrival at'] - env.now)

        cargo_type = train_info['cargo type']
        terminal_id = train_info['terminal_id']
        terminal = train_info['terminal']

        if cargo_type == 'Container':
            racks = train_loading_racks_ctr[terminal_id]
            cargo_yard = port_yard_container_terminals[terminal_id]
        elif cargo_type == 'Liquid':
            racks = train_loading_racks_liquid[terminal_id]
            cargo_yard = port_tanks_liquid_terminals[terminal_id]

            if cargo_yard.level >= 0.8 * cargo_yard.capacity:
                if terminal in liq_terminals_with_pipeline_sink:
                    Pipeline(run_id, env, cargo_yard, mode='sink',
                             rate=constants.PIPELINE_RATE)
                    with open(f'.{run_id}/logs/force_action.txt', 'a') as f:
                        f.write(
                            f"Pipeline from train sink activated {env.now}")
                        f.write('\n')

            if cargo_yard.level <= 0.2 * cargo_yard.capacity:
                if terminal in liq_terminals_with_pipeline_source:
                    Pipeline(run_id, env, cargo_yard, mode='source',
                             rate=constants.PIPELINE_RATE)
                    with open(f'.{run_id}/logs/force_action.txt', 'a') as f:
                        f.write(
                            f"Pipeline from train source activated {env.now}")
                        f.write('\n')

        elif cargo_type == 'DryBulk':
            racks = train_loading_racks_dk[terminal_id]
            cargo_yard = port_silos_drybulk_terminals[terminal_id]

        car_amount = train_info['car amount']
        cargo_transfer_rate = train_info['cargo transfer rate']

        transfer_amount = train_info['total transfer cargo']
        import_bool = train_info['import']
        export_bool = train_info['export']

        Train(env, train_id, terminal_id, car_amount, cargo_transfer_rate, racks,
              cargo_yard, train_events, transfer_amount, import_bool, export_bool, cargo_type)


def data_logger(run_id, env, pilots_tugs_data, day_pilots, night_pilots, tugboats):
    """
    This function logs the availability of pilots and tugboats at the port over time.
    It creates a DataFrame to store the time, number of day pilots, night pilots, and tugboats available at each time step.
    Args:
        run_id (str): The unique identifier for the simulation run.
        env (simpy.Environment): The simulation environment.
        pilots_tugs_data (pd.DataFrame): DataFrame to store pilots and tugboats data.
        day_pilots (simpy.Resource): Resource representing day pilots.
        night_pilots (simpy.Resource): Resource representing night pilots.
        tugboats (simpy.Resource): Resource representing tugboats.
    Yields:
        simpy.Timeout: A timeout event to log the data at each time step.
    Returns:
        None
    """

    while env.now < constants.SIMULATION_TIME-1:

        new_row = pd.DataFrame([{
            'Time': env.now,
            'Day Pilots': day_pilots.level,
            'Night Pilots': night_pilots.level,
            'Tugboats': tugboats.level
        }])

        pilots_tugs_data = pd.concat(
            [pilots_tugs_data, new_row], ignore_index=True)
        yield env.timeout(1)

    pilots_tugs_data.to_csv(
        f'.{run_id}/logs/pilots_tugs_data.csv', index=False)
