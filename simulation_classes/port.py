"""
This module defines the port resources and their initialization for a port simulation.
It includes classes for containers, cranes, berths, pipelines, conveyors, and their respective initialization methods.
This module is part of a larger port simulation system that models the operations of a port, including handling of containers, liquid bulk, and dry bulk cargo.
"""
import simpy
import random
import pandas as pd
import constants

from simulation_handler.helpers import get_value_by_terminal, get_values_by_terminal_random_sample, normal_random_with_limit, get_value_from_terminal_tuple

# initial terminal storage levels set as 50%
prefill_yard_per = 0.5
prefill_tank_per = 0.5
prefill_silo_per = 0.5
NUM_PILOTS_DAY = constants.NUM_PILOTS_DAY
NUM_PILOTS_NIGHT = constants.NUM_PILOTS_NIGHT
NUM_TUGBOATS = constants.NUM_TUGBOATS


class Container:
    """Class representing a container in the port yard.
    Args:
        id (str): Unique identifier for the container.
        width (int): Width of the container.
    """

    def __init__(self, id, width):
        self.id = id
        self.width = width


class Crane:
    """Class representing a crane in a container berth.
    Args:
        id (str): Unique identifier for the crane.
        width (int): Width of the crane.
        crane_transfer_rate (float): Rate at which the crane can transfer containers.
    """

    def __init__(self, id, width, crane_transfer_rate):
        self.id = id
        self.width = width
        self.crane_transfer_rate = crane_transfer_rate


class Berth_Ctr:
    """Class representing a container berth with multiple cranes.
    Args:
        env (simpy.Environment): The simulation environment.
        id (str): Unique identifier for the berth.
        cranes_per_berth (int): Number of cranes in the berth.
        crane_transfer_rate (float): Rate at which each crane can transfer containers.
    """

    def __init__(self, env, id, cranes_per_berth, crane_transfer_rate):
        self.id = id
        self.cranes = simpy.Store(env, capacity=cranes_per_berth)
        self.crane_transfer_rate = crane_transfer_rate
        for i in range(cranes_per_berth):
            self.cranes.put(
                Crane(id=f"{id}.{i}", width="width", crane_transfer_rate=crane_transfer_rate))


class Pipeline:
    """Class representing a pipeline in a liquid berth.
    Args:
        env (simpy.Environment): The simulation environment.
        id (str): Unique identifier for the pipeline.
        pump_rate_per_pipeline (float): Rate at which the pipeline can pump liquid.
    """

    def __init__(self, env, id, pump_rate_per_pipeline):
        self.env = env
        self.id = id
        self.pump_rate_per_pipeline = pump_rate_per_pipeline


class Berth_Liq:
    """Class representing a liquid berth with multiple pipelines.
    Args:
        env (simpy.Environment): The simulation environment.
        id (str): Unique identifier for the berth.
        piplines_per_berth (int): Number of pipelines in the berth.
        pump_rate_per_pipeline (float): Rate at which each pipeline can pump liquid.
    """

    def __init__(self, env, id, piplines_per_berth, pump_rate_per_pipeline):
        self.id = id
        self.pipelines = simpy.Store(env, capacity=piplines_per_berth)
        for i in range(piplines_per_berth):
            self.pipelines.put(Pipeline(
                env=env, id=f"{id}.{i}", pump_rate_per_pipeline=pump_rate_per_pipeline))


class Conveyor:
    """
    Class representing a conveyor in a dry bulk berth.
     Args:
        env (simpy.Environment): The simulation environment.
        id (str): Unique identifier for the conveyor.
        conveyor_rate_per_conveyor (float): Rate at which the conveyor can transfer dry bulk.
    """

    def __init__(self, env, id, conveyor_rate_per_conveyor):
        self.env = env
        self.id = id
        self.conveyor_rate_per_conveyor = conveyor_rate_per_conveyor


class Berth_DryBulk:
    """Class representing a dry bulk berth with multiple conveyors.
    Args:
        env (simpy.Environment): The simulation environment.
        id (str): Unique identifier for the berth.
        conveyors_per_berth (int): Number of conveyors in the berth.
        conveyor_rate_per_conveyor (float): Rate at which each conveyor can transfer dry bulk.
    """

    def __init__(self, env, id, conveyors_per_berth, conveyor_rate_per_conveyor):
        self.id = id
        self.conveyors = simpy.Store(env, capacity=conveyors_per_berth)
        for i in range(conveyors_per_berth):
            self.conveyors.put(Conveyor(
                env=env, id=f"{id}.{i}", conveyor_rate_per_conveyor=conveyor_rate_per_conveyor))


def create_resources(terminal_data, run_id, terminal_data_df, num_terminals, env, seed):
    """
    Create and initialize resources for the port simulation based on terminal data.
    This function sets up yards, berths, tanks, silos, loading bays, truck gates, train loading racks,
    and pilot/tugboat resources for each terminal type (Container, Liquid, DryBulk).
    Args:
        terminal_data (dict): Dictionary containing terminal data.
        run_id (str): Unique identifier for the simulation run.
        terminal_data_df (pd.DataFrame): DataFrame containing terminal data.
        num_terminals (tuple): Tuple containing the number of terminals for each cargo type (Container, Liquid, DryBulk).
        env (simpy.Environment): The simulation environment.
        seed (int): Random seed for reproducibility.
    Returns:
        terminal_resouces (list): List of resources created for each terminal type. This includes yards, berths, tanks, silos, loading bays, truck gates, train loading racks, pilots, tugboats, and channel scheduler.
    """
    num_container_terminals, num_liquid_terminals, num_drybulk_terminals = num_terminals

    #############################
    # Container terminal resources
    #############################

    random.seed(seed)
    # Yard
    port_yard_container_terminals = [
        simpy.Store(env, capacity=get_value_by_terminal(
            terminal_data, 'Container', idx + 1, 'storage volume'))
        for idx in range(num_container_terminals)
    ]
    for idx in range(num_container_terminals):
        initial_capacity = int(prefill_yard_per * get_value_by_terminal(
            terminal_data, 'Container', idx + 1, 'storage volume'))
        for i in range(initial_capacity):
            port_yard_container_terminals[idx].put(
                Container(id=f"initial_{i}", width=0))

    port_chassis_container_terminals = [
        simpy.Store(env, capacity=get_value_by_terminal(terminal_data, 'Container', idx +
                    1, 'truck bays/chassis'))  # Setting capacity to infinity or a large value
        for _ in range(num_container_terminals)
    ]

    # Initialize each store with the desired number of items
    for idx, store in enumerate(port_chassis_container_terminals):
        init_items = get_value_by_terminal(
            terminal_data, 'Container', idx + 1, 'truck bays/chassis')
        for _ in range(init_items):
            store.put(1)

    # Berths
    port_berths_container_terminals = [
        simpy.Store(env, capacity=get_value_by_terminal(
            terminal_data, 'Container', idx + 1, 'Berths'))
        for idx in range(num_container_terminals)
    ]

    # Add cranes to the container berths
    for idx, terminal in enumerate(port_berths_container_terminals):
        num_cranes_list = get_values_by_terminal_random_sample(
            terminal_data_df, 'Container', idx+1, 'transfer units per berth', num_samples=get_value_by_terminal(terminal_data, 'Container', idx + 1, 'Berths'), seed=seed)

        for i in range(get_value_by_terminal(terminal_data, 'Container', idx + 1, 'Berths')):
            terminal.put(Berth_Ctr(env=env, id=i, cranes_per_berth=num_cranes_list[i], crane_transfer_rate=get_value_by_terminal(
                terminal_data, 'Container', idx + 1, 'transfer rate per unit')))
            # print(f"Container Terminal {idx+1} Berth {i} has {num_cranes_list[i]} cranes")

    #############################
    # Liquid bulk terminal resources
    #############################

    # Tanks
    tank_capacity = [get_value_by_terminal(
        terminal_data, 'Liquid', idx + 1, 'storage volume') for idx in range(num_liquid_terminals)]
    port_tanks_liquid_terminals = [simpy.Container(env, capacity=capacity, init=int(
        prefill_tank_per*capacity)) for capacity in tank_capacity]

    #  Loading bays
    port_loading_bays_liquid_terminals = [simpy.Resource(env, capacity=get_value_by_terminal(
        terminal_data, 'Liquid', idx + 1, 'truck bays/chassis')) for idx in range(num_liquid_terminals)]

    # Berths
    port_berth_liquid_terminals = [
        simpy.Store(env, capacity=get_value_by_terminal(
            terminal_data, 'Liquid', idx + 1, 'Berths'))
        for idx in range(num_liquid_terminals)
    ]

    #  Add pipelines to the liquid berths
    for idx, terminal in enumerate(port_berth_liquid_terminals):
        num_pipelines_list = get_values_by_terminal_random_sample(
            terminal_data_df, 'Liquid', idx+1, 'transfer units per berth', num_samples=get_value_by_terminal(terminal_data, 'Liquid', idx + 1, 'Berths'), seed=seed)
        for i in range(get_value_by_terminal(terminal_data, 'Liquid', idx + 1, 'Berths')):
            terminal.put(Berth_Liq(env=env, id=i, piplines_per_berth=num_pipelines_list[i], pump_rate_per_pipeline=get_value_by_terminal(
                terminal_data, 'Liquid', idx + 1, 'transfer rate per unit')))
            # print(f"Liquid Terminal {idx+1} Berth {i} has {num_pipelines_list[i]} pipelines")

    #############################
    # Dry bulk terminal resources
    #############################

    # Silos
    silo_capacity = [get_value_by_terminal(
        terminal_data, 'DryBulk', idx + 1, 'storage volume') for idx in range(num_drybulk_terminals)]
    port_silos_drybulk_terminals = [simpy.Container(env, capacity=capacity, init=int(
        prefill_silo_per*capacity)) for capacity in silo_capacity]

    # Drybulk bays
    port_drybulk_bays_drybulk_terminals = [simpy.Resource(env, capacity=get_value_by_terminal(
        terminal_data, 'DryBulk', idx + 1, 'truck bays/chassis')) for idx in range(num_drybulk_terminals)]

    # Berths
    port_berth_drybulk_terminals = [
        simpy.Store(env, capacity=get_value_by_terminal(
            terminal_data, 'DryBulk', idx + 1, 'Berths'))
        for idx in range(num_drybulk_terminals)
    ]

    # Conveyors
    for idx, terminal in enumerate(port_berth_drybulk_terminals):
        num_conveyors_list = get_values_by_terminal_random_sample(
            terminal_data_df, 'DryBulk', idx+1, 'transfer units per berth', num_samples=get_value_by_terminal(terminal_data, 'DryBulk', idx + 1, 'Berths'), seed=seed)
        for i in range(get_value_by_terminal(terminal_data, 'DryBulk', idx + 1, 'Berths')):
            terminal.put(Berth_DryBulk(env=env, id=i, conveyors_per_berth=num_conveyors_list[i], conveyor_rate_per_conveyor=get_value_by_terminal(
                terminal_data, 'DryBulk', idx + 1, 'transfer rate per unit')))
            # print(f"Drybulk Terminal {idx+1} Berth {i} has {num_conveyors_list[i]} conveyors")

    #############################
    # Truck gates
    #############################

    truck_gates_ctr = [simpy.Resource(env, capacity=get_value_by_terminal(
        terminal_data, 'Container', idx + 1, 'truck gates')) for idx in range(num_container_terminals)]
    truck_gates_liquid = [simpy.Resource(env, capacity=get_value_by_terminal(
        terminal_data, 'Liquid', idx + 1, 'truck gates')) for idx in range(num_liquid_terminals)]
    truck_gates_dk = [simpy.Resource(env, capacity=get_value_by_terminal(
        terminal_data, 'DryBulk', idx + 1, 'truck gates')) for idx in range(num_drybulk_terminals)]

    #############################
    # Train resources
    #############################

    # Loading racks (for loading operations)
    train_loading_racks_ctr = [
        simpy.Resource(env, capacity=get_value_by_terminal(
            terminal_data, 'Container', idx + 1, 'train car loading bays'))
        for idx in range(num_container_terminals)
    ]
    train_loading_racks_liquid = [
        simpy.Resource(env, capacity=get_value_by_terminal(
            terminal_data, 'Liquid', idx + 1, 'train car loading bays'))
        for idx in range(num_liquid_terminals)
    ]
    train_loading_racks_dk = [
        simpy.Resource(env, capacity=get_value_by_terminal(
            terminal_data, 'DryBulk', idx + 1, 'train car loading bays'))
        for idx in range(num_drybulk_terminals)
    ]

    #############################
    # Pilot and tugboat resources
    #############################

    num_pilots_day = int(normal_random_with_limit(
        NUM_PILOTS_DAY[0], NUM_PILOTS_DAY[1], seed))
    num_pilots_night = int(normal_random_with_limit(
        NUM_PILOTS_NIGHT[0], NUM_PILOTS_NIGHT[1], seed))
    num_tugboats = int(normal_random_with_limit(
        NUM_TUGBOATS[0], NUM_TUGBOATS[1], seed))

    report_path = f".{run_id}/logs/final_report.txt"
    with open(report_path, 'a') as f:
        f.write(f"Num pilots in day shift: {num_pilots_day}\n")
        f.write(f"Num pilots in night shift: {num_pilots_night}\n")
        f.write(f"Num tugboats: {num_tugboats}\n\n")

    day_pilots = simpy.Container(
        env, init=num_pilots_day, capacity=num_pilots_day)
    night_pilots = simpy.Container(
        env, init=num_pilots_night, capacity=num_pilots_night)
    tugboats = simpy.Container(env, init=num_tugboats, capacity=num_tugboats)

    #############################
    # Channel scheduler
    #############################

    # Only one ship can be scheduled at a time
    channel_scheduer = simpy.Resource(env, capacity=1)

    #############################
    # Combines and returns all terminal resources for all cargo types
    #############################

    terminal_resouces = [port_berths_container_terminals, port_yard_container_terminals, port_berth_liquid_terminals,
                         port_tanks_liquid_terminals, port_berth_drybulk_terminals, port_silos_drybulk_terminals,
                         port_loading_bays_liquid_terminals, port_drybulk_bays_drybulk_terminals, port_chassis_container_terminals,
                         truck_gates_ctr, truck_gates_liquid, truck_gates_dk, train_loading_racks_ctr, train_loading_racks_liquid,
                         train_loading_racks_dk, day_pilots, night_pilots, tugboats, channel_scheduer]
    return terminal_resouces
