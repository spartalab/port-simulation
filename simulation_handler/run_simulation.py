"""
This module runs the port simulation, generating ships, trucks, and trains,
creating resources, and executing the simulation until the specified time.
"""

import gc
import psutil
import time
import os
import shutil

import simpy
import pandas as pd

from simulation_handler.helpers import clear_logs, clear_env
from simulation_handler.preprocess import generate_ships, generate_trucks, generate_trains, get_piplines_import
from simulation_handler.generators import ship_generator, truck_generator, train_generator, data_logger
from simulation_handler.helpers import clean_data, create_terminal_data_cache, create_terminal_tuple_cache
from simulation_classes.port import create_resources
from simulation_classes.channel import Channel
from simulation_analysis.resource_utilization import bottleneckAnalysis
from simulation_analysis.whatif_scenarios import *
from simulation_analysis.results import plot_channel, gen_logs_and_plots
import constants


def print_memory_usage():
    """Print the current memory usage of the process."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024**2:.2f} MB (RSS), {memory_info.vms / 1024**2:.2f} MB (VMS)")

def run_simulation(seed):
    """
    Run the port simulation with the given seed.
    This function initializes the simulation environment, generates ships, trucks, and trains,
    creates resources, and runs the simulation until the specified time.
    Args:
        seed (int): Random seed for reproducibility.
    """
    print("Processing seed: ", seed)
    start_time = time.time()
    run_id = f"Results_{seed}_{int(constants.NUM_MONTHS)}_months_{constants.ARRIVAL_INCREASE_FACTOR}"
    clear_logs(run_id)
    ship_logs = []

    print("Preprocessing data...")
    terminal_data_df = clean_data(constants.directory)
    terminal_data = create_terminal_data_cache(terminal_data_df, run_id, seed)
    terminal_tuple_cache = create_terminal_tuple_cache(terminal_data_df, run_id, seed)

    df = pd.DataFrame(terminal_data.items(), columns=['Key', 'Amount Allocated'])
    df[['Terminal type', 'Terminal Number', 'Resource Allocated']] = pd.DataFrame(df['Key'].tolist(), index=df.index)
    df = df.drop(columns=['Key'])
    df = df[['Terminal type', 'Terminal Number', 'Resource Allocated', 'Amount Allocated']]
    df.to_csv(f'.{run_id}/logs/terminal_data_cache.csv', index=False)

    num_container_terminals = terminal_data_df[(terminal_data_df['Cargo'] == 'Container')]['Terminal'].nunique()
    num_liquid_terminals = terminal_data_df[(terminal_data_df['Cargo'] == 'Liquid')]['Terminal'].nunique()
    num_drybulk_terminals = terminal_data_df[(terminal_data_df['Cargo'] == 'DryBulk')]['Terminal'].nunique()
    num_terminals_list = [num_container_terminals, num_liquid_terminals, num_drybulk_terminals]

    print("Generating ships...")
    generate_ships(run_id, num_terminals_list, seed)
    print("Generating trucks...")
    generate_trucks(run_id, num_terminals_list, terminal_data_df, terminal_tuple_cache, seed)
    print("Generating trains...")
    generate_trains(run_id, num_terminals_list, terminal_data, terminal_data_df, terminal_tuple_cache, seed)

    liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink = get_piplines_import(num_terminals_list, terminal_data)

    plot_channel(run_id)

    print("Creating resources...")
    env = simpy.Environment()
    events = []
    train_events = {}
    channel_events = []
    channel_logs = [] 
    chassis_bays_utilization = {}
    for terminal_type in ["Container", "Liquid", "DryBulk"]:
        chassis_bays_utilization[terminal_type] = {}
        for terminal_id in range(1, num_terminals_list[["Container", "Liquid", "DryBulk"].index(terminal_type)] + 1):
            chassis_bays_utilization[terminal_type][terminal_id] = []

    terminal_resouces = create_resources(terminal_data, run_id, terminal_data_df, num_terminals_list, env, seed)

    if constants.MODEL_HURRICANE:
         model_hurricane(env, terminal_resouces, num_terminals_list, terminal_data, run_id, seed)

    ship_data_df = pd.read_csv(f".{run_id}/logs/ship_data.csv")
    ship_data = ship_data_df.to_dict(orient="index")
    SIMULATION_TIME = constants.SIMULATION_TIME
    NUM_CHANNEL_SECTIONS = constants.NUM_CHANNEL_SECTIONS
    CHANNEL_SAFETWOWAY = constants.CHANNEL_SAFETWOWAY
    port_berths_container_terminals, port_yard_container_terminals, port_berth_liquid_terminals, port_tanks_liquid_terminals, \
    port_berth_drybulk_terminals, port_silos_drybulk_terminals, port_loading_bays_liquid_terminals, port_drybulk_bays_drybulk_terminals, \
    port_chassis_container_terminals, truck_gates_ctr, truck_gates_liquid, truck_gates_dk, train_loading_racks_ctr, train_loading_racks_liquid, \
                         train_loading_racks_dk, day_pilots, night_pilots, tugboats, channel_scheduer = terminal_resouces

    # Creating channel...
    SHIPS_IN_ANCHORAGE = [0,0,0] 
    SHIPS_IN_CHANNEL = []
    SHIPS_IN_CHANNEL.append(0) 
    if constants.MODEL_FOG:
        turnoffTime = {"switch": "channel_closed", "closed_between": constants.FOG_CLOSURES}
    else:
        turnoffTime = {"switch": "channel_open"}

    channel = Channel(ship_logs, env, NUM_CHANNEL_SECTIONS, SIMULATION_TIME, CHANNEL_SAFETWOWAY, channel_events, channel_logs, day_pilots, night_pilots, tugboats, turnoffTime, channel_scheduer, seed)

    # Starting simulation...
    ship_proc = env.process(ship_generator(run_id, env, chassis_bays_utilization, port_berths_container_terminals, port_yard_container_terminals, port_berth_liquid_terminals, port_tanks_liquid_terminals, 
                            port_berth_drybulk_terminals, port_silos_drybulk_terminals, channel, day_pilots, night_pilots, tugboats, events, ship_logs, channel_events, channel_logs, SHIPS_IN_ANCHORAGE, SHIPS_IN_CHANNEL, ship_data, terminal_data, liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink))
    truck_proc = env.process(truck_generator(run_id, liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink, chassis_bays_utilization, env, terminal_tuple_cache, port_tanks_liquid_terminals, port_yard_container_terminals, port_silos_drybulk_terminals, port_loading_bays_liquid_terminals, port_drybulk_bays_drybulk_terminals, port_chassis_container_terminals, truck_gates_ctr, truck_gates_liquid, truck_gates_dk, events, seed, terminal_data))
    train_proc = env.process(train_generator(run_id, liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink, env, terminal_tuple_cache, train_loading_racks_ctr, train_loading_racks_liquid, train_loading_racks_dk, train_events, port_tanks_liquid_terminals, port_yard_container_terminals, port_silos_drybulk_terminals, seed))

    pilots_tugs_data = pd.DataFrame(columns=['Time', 'Day Pilots', 'Night Pilots', 'Tugboats'])
    data_taker_proc = env.process(data_logger(run_id, env, pilots_tugs_data, day_pilots, night_pilots, tugboats))

    env.run(until=SIMULATION_TIME)

    clear_env(env, ship_proc, truck_proc, train_proc, data_taker_proc)
    gen_logs_and_plots(run_id, ship_logs, events, chassis_bays_utilization, num_terminals_list, train_events, channel_logs, channel_events, channel, animate=False)
    bottleneckAnalysis(run_id)
    
    # Free memory
    # print_memory_usage()

    del env, ship_proc, truck_proc, terminal_resouces, terminal_data, events, channel_events, channel_logs
    del port_berths_container_terminals, port_yard_container_terminals, port_berth_liquid_terminals, port_tanks_liquid_terminals
    del port_berth_drybulk_terminals, port_silos_drybulk_terminals, port_loading_bays_liquid_terminals, port_drybulk_bays_drybulk_terminals
    del port_chassis_container_terminals, truck_gates_ctr, truck_gates_liquid, truck_gates_dk, day_pilots, night_pilots, tugboats
    gc.collect()

    # remove truck pickle file from "/.Results*" output folder (saves hard drive space)
    if os.path.exists(f".{run_id}/logs/truck_data.pkl"):
        os.remove(f".{run_id}/logs/truck_data.pkl")
    else:
        print("Truck pkl file does not exist")
        pass 

    # revome the availablity folder 
    dir_path = f".{run_id}/logs/availability"

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path) 
    else:
        print("Availability folder does not exist")

    # close all open files and plots
    plt.close('all')
    
