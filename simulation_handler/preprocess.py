"""
This module generates ship, truck, and train data for a maritime simulation.
It includes functions to initialize random number generators, generate truncated exponential distributions,
select terminals based on ship type and size, and create ship and truck data based on predefined probabilities.
It also provides functions to fill ship details, generate random sizes, and create ship and truck data dictionaries.
"""
import random
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, expon
from tqdm import tqdm

from constants import *
from simulation_handler.helpers import normal_random_with_sd, normal_random_with_limit, get_value_by_terminal, get_values_by_terminal_random_sample, save_warning
from simulation_handler.helpers import is_daytime

global NUM_TRUNCATION
NUM_TRUNCATION = {"Container": 0, "Liquid": 0, "DryBulk": 0}

def initialize_rng(seed):
    """
    Initialize random state objects for repeatable randomness across runs.
    This function sets up two random number generators: one using Python's built-in `random` module
    and another using NumPy's random number generator. This allows for consistent random behavior
    across different runs of the simulation, which is crucial for debugging and testing.
    Args:
        seed (int): The seed value to initialize the random number generators.
    Returns:
        None
    """
    global rng_random, rng_numpy
    rng_random = random.Random(seed)
    rng_numpy = np.random.default_rng(seed)

def truncated_exponential(a, b, scale, ship_type, ship_terminal):
    """ 
    Generate a truncated exponential random variable.   
    This function generates a random variable from an exponential distribution
    truncated to the interval [a, b]. If the generated value exceeds b, it is set to b.
    Args:
        a (float): The lower bound of the truncation interval.
        b (float): The upper bound of the truncation interval.
        scale (float): The scale parameter of the exponential distribution.
        ship_type (str): Type of ship ('Container', 'Liquid', 'DryBulk').
        ship_terminal (int): Terminal number where the ship is located.
    Returns:
        float: A random variable from the truncated exponential distribution.
    """
    global NUM_TRUNCATION
    candidate = rng_numpy.exponential(scale)
    if candidate >= b:
        if ship_type == 'Container':
                NUM_TRUNCATION[ship_type] += 1
        elif ship_type == 'Liquid':
                NUM_TRUNCATION[ship_type] += 1
        elif ship_type == 'DryBulk':
                NUM_TRUNCATION[ship_type] += 1
        candidate = b

    return candidate

def truncated_exponential_advanced(a, b, scale, ship_type, ship_terminal):
    """ 
    Generate a truncated exponential random variable with advanced truncation.
    This function generates a random variable from an exponential distribution
    truncated to the interval [a, b]. If the generated value exceeds b, it is set to b.
    Args:
        a (float): The lower bound of the truncation interval.
        b (float): The upper bound of the truncation interval.
        scale (float): The scale parameter of the exponential distribution.
        ship_type (str): Type of ship ('Container', 'Liquid', 'DryBulk').
        ship_terminal (int): Terminal number where the ship is located.
    Returns:
        float: A random variable from the truncated exponential distribution.
    """

    cdf_a = 1 - np.exp(-a / scale)
    cdf_b = 1 - np.exp(-b / scale)
    u = rng_numpy.uniform(cdf_a, cdf_b)
    samples = -scale * np.log(1 - u) 
    return samples

def select_terminal(ship_type, ship_beam, num_terminals, seed):
    """
    Select terminal based on probability based on number of berths in that terminal to the total number of berths in all terminal of that type.
    This function selects a terminal for a ship based on its type and beam size.
    If the ship is larger than a certain beam size, it selects from a limited set of terminals.
    Args:
        ship_type (str): Type of ship ('Container', 'Liquid', 'DryBulk').
        ship_beam (float): Beam size of the ship.
        num_terminals (list): List containing the number of terminals for each type of ship [num_container_terminals, num_liquid_terminals, num_drybulk_terminals].
        seed (int): Random seed for reproducibility.
    Returns:
        int: Selected terminal number based on the ship type and beam size.
    """
    random.seed(seed)
    num_container_terminals, num_liquid_terminals, num_drybulk_terminals = num_terminals
    if ship_type == "Container":
        if ship_beam > MAX_BEAM_SMALL_SHIP:
            total_berths = sum(BERTHS_CTR_TERMINAL)
            berth_probs = [berth/total_berths for berth in BERTHS_CTR_TERMINAL][:NO_LARGE_SHIP_BEYOND_THIS_TERMINAL_CTR]
            berth_probs = [x/sum(berth_probs) for x in berth_probs]
            return random.choices(range(1, NO_LARGE_SHIP_BEYOND_THIS_TERMINAL_CTR+1), berth_probs)[0]
        else:
            total_berths = sum(BERTHS_CTR_TERMINAL)
            berth_probs = [berth/total_berths for berth in BERTHS_CTR_TERMINAL]
            return random.choices(range(1, num_container_terminals+1), berth_probs)[0]
    elif ship_type == "Liquid":
        if ship_beam > MAX_BEAM_SMALL_SHIP:
            total_berths = sum(BERTHS_LIQ_TERMINAL)
            berth_probs = [berth/total_berths for berth in BERTHS_LIQ_TERMINAL][:NO_LARGE_SHIP_BEYOND_THIS_TERMINAL_LIQ]
            berth_probs = [x/sum(berth_probs) for x in berth_probs]
            return random.choices(range(1, NO_LARGE_SHIP_BEYOND_THIS_TERMINAL_LIQ+1), berth_probs)[0]
        else:
            total_berths = sum(BERTHS_LIQ_TERMINAL)
            berth_probs = [berth/total_berths for berth in BERTHS_LIQ_TERMINAL]
            return random.choices(range(1, num_liquid_terminals+1), berth_probs)[0]
    elif ship_type == "DryBulk":
        if ship_beam > MAX_BEAM_SMALL_SHIP:
            total_berths = sum(BERTH_DRYBULK_TERMINAL)
            berth_probs = [berth/total_berths for berth in BERTH_DRYBULK_TERMINAL][:NO_LARGE_SHIP_BEYOND_THIS_TERMINAL_DRYBULK]
            berth_probs = [x/sum(berth_probs) for x in berth_probs]
            return random.choices(range(1, NO_LARGE_SHIP_BEYOND_THIS_TERMINAL_DRYBULK+1), berth_probs)[0]
        else:
            total_berths = sum(BERTH_DRYBULK_TERMINAL)
            berth_probs = [berth/total_berths for berth in BERTH_DRYBULK_TERMINAL]
            return random.choices(range(1, num_drybulk_terminals+1), berth_probs)[0]

def assign_random_size(ship_type, probabilities):
    """
    Assign a random size to a ship based on its type.
    This function uses predefined probabilities to assign a size category ('Small', 'Medium', 'Large') to a ship.
    Args:
        ship_type (str): Type of ship ('Container', 'DryBulk', 'Liquid').
        probabilities (dict): Dictionary containing size probabilities for each ship type.
    Returns:
        str: Assigned size category ('Small', 'Medium', 'Large') or NaN if ship type is invalid.
    """ 
    if ship_type == 'Container':
        return rng_numpy.choice(['Small', 'Medium', 'Large'], p=probabilities["container"])
    elif ship_type == 'DryBulk':
        return rng_numpy.choice(['Small', 'Medium', 'Large'], p = probabilities["drybulk"])
    elif ship_type == 'Liquid':
        return rng_numpy.choice(['Small', 'Medium', 'Large'], p = probabilities["liquid"])
    else:
        return np.nan

def generate_ship_data(ship_type, NUM_SHIPS, probabilities):
    """ 
    Generate ship data for a specific ship type.    
    This function generates a dictionary of ship data for a given ship type, including ship ID, direction, ship type, and size.
    Args:
        ship_type (str): Type of ship ('Container', 'DryBulk', 'Liquid').
        NUM_SHIPS (dict): Dictionary containing the number of ships for each ship type.
    Returns:
        dict: Dictionary containing ship data for the specified ship type.
    """
    return {
        f"{i}": {
            "ship_id": i,
            "direction": "in",
            "ship_type": (ship_type),
            "Size" : assign_random_size(ship_type, probabilities),
        } for i in range(NUM_SHIPS[ship_type])
    }

def fill_ship_details(row, vessel_sizes, seed):
    """ Fill ship details such as length, beam, draft, tonnage, pilots, and tugboats based on the ship type and size.
    This function retrieves the average and standard deviation values for ship dimensions and tonnage from a DataFrame
    containing vessel size data. It then samples from a normal distribution to generate realistic ship dimensions and tonnage.
    Args:
        row (pd.Series): A row from the ship data DataFrame containing ship type and size.
        vessel_sizes (pd.DataFrame): DataFrame containing vessel size information with average and standard deviation values.
        seed (int): Random seed for reproducibility.
    Returns:
        pd.Series: A Series containing the generated ship details: length, beam, draft, tonnage, pilots, and tugboats.
    """
    size_data = vessel_sizes[(vessel_sizes['ship_type'] == row['ship_type']) & (vessel_sizes['Size'] == row['Size'])]
    
    if not size_data.empty:
        # Sample from a normal distribution using average and standard deviation
        length = normal_random_with_sd(size_data['Avg_Length'].values[0], size_data['Std_Length'].values[0], rng_random.randint(0, 100000), scale_factor = 2)
        beam = normal_random_with_sd(size_data['Avg_Beam'].values[0], size_data['Std_Beam'].values[0], rng_random.randint(0, 100000), scale_factor = 2)
        draft = normal_random_with_sd(size_data['Avg_Draft'].values[0], size_data['Std_Draft'].values[0], rng_random.randint(0, 100000), scale_factor = 2)
        tonnage = normal_random_with_sd(size_data['Avg_Tonnage'].values[0], size_data['Std_Tonnage'].values[0], rng_random.randint(0, 100000), scale_factor = 3)
        if row['ship_type'] == 'Container':
            tonnage =  (1 - NON_CARGO_DEAD_WEIGHT_PERCENT_CTR) * tonnage
        elif row['ship_type'] == 'DryBulk':
            tonnage =  (1 - NON_CARGO_DEAD_WEIGHT_PERCENT_DK) * tonnage
        elif row['ship_type'] == 'Liquid':
            tonnage =  (1 - NON_CARGO_DEAD_WEIGHT_PERCENT_LIQ) * tonnage
        else:
            raise ValueError("Invalid ship type")
        if tonnage < 0:
            print(f"Negative tonnage: {tonnage}, ship type: {row['ship_type']}, size: {row['Size']}, length: {length}, beam: {beam}, draft: {draft}")
        if row['ship_type'] == 'Liquid':
            tonnage = tonnage * rng_random.choice(LIQUID_CONVERSION_FACTORS)
        elif row['ship_type'] == 'DryBulk':
            tonnage = tonnage
        else: 
            tonnage = int(tonnage * rng_random.choice(CONTAINER_CONVERSION_FACTORS))
        pilots = int(rng_numpy.uniform(size_data['min_pilots'].values[0], size_data['max_pilots'].values[0]+1))
        tugboats = int(rng_numpy.uniform(size_data['min_tugs'].values[0], size_data['max_tugs'].values[0]+1))
        return pd.Series([length, beam, draft, tonnage, pilots, tugboats])
    else:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

def generate_truck_data(truck_type, NUM_TRUCKS):
    """
    Generate truck data for a specific truck type.
    This function generates a dictionary of truck data for a given truck type, including truck ID, direction, and truck type.
    Args:
        truck_type (str): Type of truck ('Container', 'DryBulk', 'Liquid').
        NUM_TRUCKS (dict): Dictionary containing the number of trucks for each truck type.
    Returns:
        dict: Dictionary containing truck data for the specified truck type.
    """
    if truck_type == 'Container':
        return {
            f"{i}": {
                "truck_id": i,
                "direction": "in",
                "truck_type": (truck_type),
            } for i in range(NUM_TRUCKS[truck_type])
        }
    elif truck_type == 'DryBulk':
        return {
            f"{i}": {
                "truck_id": i,
                "direction": "in",
                "truck_type": (truck_type),
            } for i in range(NUM_TRUCKS[truck_type])
        }   
    elif truck_type == 'Liquid':
        return {
            f"{i}": {
                "truck_id": i,
                "direction": "in",
                "truck_type": (truck_type),
            } for i in range(NUM_TRUCKS[truck_type])
        }

def generate_ships(run_id, NUM_TERMINALS_LIST, seed):
    """
    Generate ship data for the simulation.
    This function creates a DataFrame containing ship data for different types of ships (Container, Liquid, DryBulk).
    It initializes the random number generator, generates ship data based on predefined probabilities,
    and assigns ship details such as length, beam, draft, tonnage, pilots, tugboats, and terminal.
    Args:
        run_id (str): Unique identifier for the simulation run.
        NUM_TERMINALS_LIST (list): List containing the number of terminals for each ship type [num_container_terminals, num_liquid_terminals, num_drybulk_terminals].
        seed (int): Random seed for reproducibility.
    Returns:
        None
    """
    vessel_size_inputs = pd.read_csv('inputs/ship_sizes.csv')  

    container_class_probs = vessel_size_inputs[vessel_size_inputs['ship_type'] == 'Container']['Fraction'].values
    liquid_class_probs = vessel_size_inputs[vessel_size_inputs['ship_type'] == 'Liquid']['Fraction'].values
    drybulk_class_probs = vessel_size_inputs[vessel_size_inputs['ship_type'] == 'DryBulk']['Fraction'].values
    probabilities = {"container": container_class_probs, "liquid": liquid_class_probs, "drybulk": drybulk_class_probs}
    
    initialize_rng(seed)
    num_container_terminals, num_liquid_terminals, num_drybulk_terminals = NUM_TERMINALS_LIST

    # Number of ships for each cargo type
    NUM_SHIPS = {
        "Container": 3 * int(SIMULATION_TIME // ((mean_interarrival_time_container)*1)), 
        "Liquid": 3 * int(SIMULATION_TIME // ((mean_interarrival_time_tanker)*1)),  
        "DryBulk": 3 * int(SIMULATION_TIME // ((mean_interarrival_time_gencargo)*1)),
    }

    SCALE_VALS_TERMINALS = {
        "Container": mean_interarrival_time_container,
        "Liquid": mean_interarrival_time_tanker,
        "DryBulk": mean_interarrival_time_gencargo
    }

    CONTAINER_SHIP_DATA, CARGO_SHIP_DATA, TANKER_SHIP_DATA = generate_ship_data('Container', NUM_SHIPS, probabilities), generate_ship_data('DryBulk', NUM_SHIPS, probabilities), generate_ship_data('Liquid', NUM_SHIPS, probabilities)

    df_tanker = pd.DataFrame(TANKER_SHIP_DATA).T
    df_cargo = pd.DataFrame(CARGO_SHIP_DATA).T
    df_container = pd.DataFrame(CONTAINER_SHIP_DATA).T

    ship_data_components = []

    for ship_type_name in ['Liquid', 'DryBulk', 'Container']:

        if ship_type_name == 'Liquid':
            df_ship = df_tanker
            num_terminals = num_liquid_terminals
        elif ship_type_name == 'DryBulk':
            df_ship = df_cargo
            num_terminals = num_drybulk_terminals
        elif ship_type_name == 'Container':
            df_ship = df_container
            num_terminals = num_container_terminals

        df_ship[['length', 'width', 'draft', 'num_container_or_liq_tons_or_dry_tons_to_load', 'pilots', 'tugboats']] = df_ship.apply(
                fill_ship_details, axis=1, vessel_sizes=vessel_size_inputs, seed=seed
            )
        df_ship[['length', 'width', 'draft', 'num_container_or_liq_tons_or_dry_tons_to_unload', 'pilots', 'tugboats']] = df_ship.apply(
            fill_ship_details, axis=1, vessel_sizes=vessel_size_inputs, seed=seed
        )
        df_ship['terminal'] = df_ship.apply(lambda row: select_terminal(row['ship_type'], row['width'], NUM_TERMINALS_LIST, seed = rng_random.randint(1, 100000000)), axis=1)
        ship_data_components.append(df_ship)
    

    ship_data = pd.concat(ship_data_components, ignore_index=True)
    ship_data_ctr = ship_data[ship_data['ship_type'] == 'Container'].copy()
    ship_data_liq = ship_data[ship_data['ship_type'] == 'Liquid'].copy()
    ship_data_drybulk = ship_data[ship_data['ship_type'] == 'DryBulk'].copy()
    ship_data_ctr["interarrival"] = ship_data_ctr.apply(lambda row: (rng_numpy.exponential(scale= SCALE_VALS_TERMINALS['Container'])), axis=1)
    ship_data_ctr['interarrival'] = pd.to_numeric(ship_data_ctr['interarrival'], errors='coerce')
    ship_data_liq["interarrival"] = ship_data_liq.apply(lambda row: (truncated_exponential_advanced(a = min_interarrival_liquid, b = max_interaarival_liq, scale= SCALE_VALS_TERMINALS['Liquid'], ship_type = 'Liquid', ship_terminal = row['terminal'])), axis=1)
    ship_data_liq['interarrival'] = pd.to_numeric(ship_data_liq['interarrival'], errors='coerce')
    ship_data_drybulk["interarrival"] = ship_data_drybulk.apply(lambda row: (rng_numpy.exponential(scale= SCALE_VALS_TERMINALS['DryBulk'])), axis=1)
    ship_data_drybulk['interarrival'] = pd.to_numeric(ship_data_drybulk['interarrival'], errors='coerce')
    ship_data_ctr['arrival'] = ship_data_ctr['interarrival'].cumsum()
    ship_data_liq['arrival'] = ship_data_liq['interarrival'].cumsum()
    ship_data_drybulk['arrival'] = ship_data_drybulk['interarrival'].cumsum()

    ship_data_ctr = ship_data_ctr.drop(columns=['interarrival'])
    ship_data_liq = ship_data_liq.drop(columns=['interarrival'])
    ship_data_drybulk = ship_data_drybulk.drop(columns=['interarrival'])

    ship_data = pd.concat([ship_data_ctr, ship_data_liq, ship_data_drybulk], ignore_index=True)
    ship_data['arrival'] = pd.to_numeric(ship_data['arrival'], errors='coerce')
    ship_data.sort_values('arrival', inplace=True)

    ship_data = ship_data[ship_data['arrival'] <= SIMULATION_TIME]
    ship_data['ship_id'] = range(1, len(ship_data) + 1)
    ship_data.reset_index(drop=True, inplace=True)
    ship_data['last_section'] = ship_data.apply(lambda row: LAST_SECTION_DICT[row['ship_type']][row['terminal']], axis=1)

    output_path = f'.{run_id}/logs/ship_data.csv'
    ship_data.to_csv(output_path, index=False)

    # for container terminals create a box plot of number of containers to load + unload for each terminal in smae plot
    container_ships = ship_data[ship_data['ship_type'] == 'Container']    
    container_ships = container_ships.copy()
    container_ships.loc[:, 'moves'] = (
        container_ships['num_container_or_liq_tons_or_dry_tons_to_load'] +
        container_ships['num_container_or_liq_tons_or_dry_tons_to_unload']
    )
    container_ships = container_ships[['terminal', 'moves']]
    plt.figure()
    container_ships.boxplot(by='terminal', column='moves')
    plt.ylim(bottom=0)
    plt.ylabel("Number of container moves")
    plt.xlabel("Container terminal")
    plt.savefig(f".{run_id}/plots/moves.pdf")
    plt.close()

    print("Total number of ships: Container {}, Liquid {}, DryBulk {}".format(len(ship_data_ctr), len(ship_data_liq), len(ship_data_drybulk)))

    # compute mean interarrival times
    liquid_data = ship_data[ship_data['ship_type'] == 'Liquid']['arrival'].diff().dropna()
    container_data = ship_data[ship_data['ship_type'] == 'Container']['arrival'].diff().dropna()
    drybulk_data = ship_data[ship_data['ship_type'] == 'DryBulk']['arrival'].diff().dropna()

    # Filter data by ship types
    print("\n\n RUN ID:", run_id)
    save_warning(run_id, f"Container terminal, Generated mean interarrival time: {round(container_data.mean(), 1)} and expected mean interarrival time: {round(mean_interarrival_time_container, 2)}")
    save_warning(run_id, f"Liquid terminal, Generated mean interarrival time: {round(liquid_data.mean(), 1)} and expected mean interarrival time: {round(mean_interarrival_time_tanker, 2)}")
    save_warning(run_id, f"DryBulk terminal, Generated mean interarrival time: {round(drybulk_data.mean(), 1)} and expected mean interarrival time: {round(mean_interarrival_time_gencargo, 2)}\n")

    save_warning(run_id, f"Minimum interarrival times: Container {min(container_data):.2e}, Liquid {min(liquid_data):.2e}, DryBulk {min(drybulk_data):.2e}\n")

    # Create a stacked bar plot of the number of ships in each terminal, split by size
    for ship_type in ['Container', 'Liquid', 'DryBulk']:
        ship_type_data = ship_data[ship_data['ship_type'] == ship_type]
        size_counts = ship_type_data.groupby(['terminal', 'Size']).size().unstack(fill_value=0)
        size_counts.plot(kind='bar', stacked=True, edgecolor="black")
        plt.xlabel("Terminal")
        plt.ylabel("Number of Ships")
        plt.title(f"Stacked Bar Plot of {ship_type} Ship Distribution by Size")
        plt.xticks(rotation=0)  # Keeps terminal labels horizontal
        plt.legend(title="Size")
        plt.tight_layout()
        plt.savefig(f".{run_id}/plots/shipDistribution/{ship_type}_ship_distribution_stacked_bar.pdf")
        plt.close()
    
    #  Plot a distribution of number of berths in each terminal. Number of berths is like sum(BERTH_DRYBULK_TERMINAL) and BERTH_DRYBULK_TERMINAL[i] for terminal i+1
    for ship_type in ['Container', 'Liquid', 'DryBulk']:
        if ship_type == 'Container':
            plt.bar(range(1, num_container_terminals+1), BERTHS_CTR_TERMINAL)
            plt.xlabel("Terminal")
            plt.ylabel("Number of berths")
            plt.title(f"Number of berths in each {ship_type} terminal")
            plt.xticks(range(1, num_container_terminals+1))
            plt.tight_layout()
            plt.savefig(f".{run_id}/plots/shipDistribution/{ship_type}_terminal_berth_distribution.pdf")
            plt.close()
        elif ship_type == 'Liquid':
            plt.bar(range(1, num_liquid_terminals+1), BERTHS_LIQ_TERMINAL)
            plt.xlabel("Terminal")
            plt.ylabel("Number of berths")
            plt.title(f"Number of berths in each {ship_type} terminal")
            plt.xticks(range(1, num_liquid_terminals+1))
            plt.tight_layout()
            plt.savefig(f".{run_id}/plots/shipDistribution/{ship_type}_terminal_berth_distribution.pdf")
            plt.close()
        elif ship_type == 'DryBulk':
            plt.bar(range(1, num_drybulk_terminals+1), BERTH_DRYBULK_TERMINAL)
            plt.xlabel("Terminal")
            plt.ylabel("Number of berths")
            plt.title(f"Number of berths in each {ship_type} terminal")
            plt.xticks(range(1, num_drybulk_terminals+1))
            plt.tight_layout()
            plt.savefig(f".{run_id}/plots/shipDistribution/{ship_type}_terminal_berth_distribution.pdf")
            plt.close()


def generate_trucks(run_id, num_terminals, terminal_data_df, terminal_tuple_cache, seed):
    """
    Generate truck data for the simulation.
    This function creates a DataFrame containing truck data for different types of trucks (Container, Liquid, DryBulk).
    It initializes the random number generator, calculates mean interarrival times for trucks at each terminal,
    generates truck data based on the mean interarrival times, and assigns terminal information to each truck.
    Args:
        run_id (str): Unique identifier for the simulation run.
        num_terminals (list): List containing the number of terminals for each truck type [num_container_terminals, num_liquid_terminals, num_drybulk_terminals].
        terminal_data_df (pd.DataFrame): DataFrame containing terminal data.
        terminal_tuple_cache (dict): Dictionary containing terminal tuple cache.
        seed (int): Random seed for reproducibility.
    Returns:
        None
    """
    start_time = time.time()
    terminal_data_df = terminal_data_df
    num_container_terminals, num_liquid_terminals, num_drybulk_terminals = num_terminals

    mean_ctr_terminal_arrival_rate = list(terminal_data_df[terminal_data_df['Cargo'] == 'Container']['truck arrival rate'].values)
    mean_liq_terminal_arrival_rate = list(terminal_data_df[terminal_data_df['Cargo'] == 'Liquid']['truck arrival rate'].values)
    mean_drybulk_terminal_arrival_rate = list(terminal_data_df[terminal_data_df['Cargo'] == 'DryBulk']['truck arrival rate'].values)

    mean_interarrival_time_container_trucks = 1 / sum(list(mean_ctr_terminal_arrival_rate))
    mean_interarrival_time_tanker_trucks = 1 / sum(list(mean_liq_terminal_arrival_rate))
    mean_interarrival_time_gencargo_trucks = 1 / sum(list(mean_drybulk_terminal_arrival_rate))
    
    NUM_CTR_TRUCKS = int(SIMULATION_TIME // ((mean_interarrival_time_container_trucks)*1))
    NUM_LIQ_TRUCKS = int(SIMULATION_TIME // ((mean_interarrival_time_tanker_trucks)*1))
    NUM_DRYBULK_TRUCKS = int(SIMULATION_TIME // ((mean_interarrival_time_gencargo_trucks)*1))
    
    NUM_TRUCKS = {
        "Container": NUM_CTR_TRUCKS,
        "Liquid": NUM_LIQ_TRUCKS,
        "DryBulk": NUM_DRYBULK_TRUCKS
    }

    CONTAINER_TRUCK_DATA = generate_truck_data('Container', NUM_TRUCKS)
    CARGO_TRUCK_DATA = generate_truck_data('DryBulk', NUM_TRUCKS)
    TANKER_TRUCK_DATA = generate_truck_data('Liquid', NUM_TRUCKS)

    df_tanker, df_cargo, df_container = [
    pd.DataFrame.from_dict(data, orient="index")
    for data in [TANKER_TRUCK_DATA, CARGO_TRUCK_DATA, CONTAINER_TRUCK_DATA]
    ]

    truck_data_components = []
    for truck_type_name in ['Liquid', 'DryBulk', 'Container']:
        if truck_type_name == 'Liquid':
            df_ship = df_tanker
            num_terminals = num_liquid_terminals
            rates = mean_liq_terminal_arrival_rate
        elif truck_type_name == 'DryBulk':
            df_ship = df_cargo
            num_terminals = num_drybulk_terminals
            rates = mean_drybulk_terminal_arrival_rate
        elif truck_type_name == 'Container':
            df_ship = df_container
            num_terminals = num_container_terminals
            rates = mean_ctr_terminal_arrival_rate

        df_truck_terminals = {}
        for terminal in range(1, num_terminals+1):
            df_truck_terminals[terminal] = df_ship.copy()
            df_truck_terminals[terminal]['terminal'] = terminal
            df_truck_terminals[terminal]['terminal_id'] = terminal-1
            df_truck_terminals[terminal]['interarrival'] = round(1 / rates[terminal-1], 8) #is a list
            df_truck_terminals[terminal]['interarrival'] = df_truck_terminals[terminal]['interarrival'].astype(float)
            df_truck_terminals[terminal]['arrival'] = df_truck_terminals[terminal]['interarrival'].cumsum()
            df_truck_terminals[terminal] = df_truck_terminals[terminal][(df_truck_terminals[terminal]['arrival'] <= SIMULATION_TIME)]
        df_truck_combined = pd.concat(df_truck_terminals.values(), ignore_index=True)
        df_truck_combined = df_truck_combined.sort_values('arrival')
        truck_data_components.append(df_truck_combined)


    truck_data = pd.concat(truck_data_components, ignore_index=True)
    truck_data['arrival'] = pd.to_numeric(truck_data['arrival'], errors='coerce')
    truck_data.sort_values('arrival', inplace=True)
    truck_data['truck_id'] = range(1, len(truck_data) + 1)
    truck_data.reset_index(drop=True, inplace=True)

    # look at cargo type and terminal and drop rows where TERMINALS_WITH_NO_TRUCKS dict 
    for cargo_type, list_of_terminals in TERMINALS_WITH_NO_TRUCKS.items():
        for terminal in list_of_terminals:
            truck_data = truck_data[~((truck_data['truck_type'] == cargo_type) & (truck_data['terminal'] == terminal))]

    # remove trucks at night using is_daytime
    truck_data = truck_data[truck_data['arrival'].apply(lambda x: is_daytime(x))]
    output_path = f'.{run_id}/logs/truck_data.csv'
    truck_data.to_pickle(f'.{run_id}/logs/truck_data.pkl')

    for terminal_type in ["Container", "Liquid", "DryBulk"]:
        terminal_data = truck_data[truck_data['truck_type'] == terminal_type]
        terminal_counts = terminal_data['terminal'].value_counts().sort_index()
        plt.bar(terminal_counts.index, terminal_counts.values)
        plt.xlabel("Terminal")
        plt.ylabel("Number of Trucks")
        plt.title(f"Number of Trucks in Each {terminal_type} Terminal")
        plt.xticks(terminal_counts.index)
        plt.tight_layout()
        plt.savefig(f".{run_id}/plots/truckDistribution/{terminal_type}_truck_distribution.pdf")
        plt.close()
        
    print("Total number of trucks", len(truck_data))

def generate_trains(run_id, num_terminals, terminal_data, terminal_data_df, terminal_tuple_cache, seed):
    """
    Generate train data for the simulation.
    This function creates a DataFrame containing train data for different types of trains (Container, Liquid, DryBulk).
    It initializes the random number generator, calculates mean interarrival times for trains at each terminal,
    generates train data based on the mean interarrival times, and assigns terminal information to each train.
    Args:
        run_id (str): Unique identifier for the simulation run.
        num_terminals (list): List containing the number of terminals for each train type [num_container_terminals, num_liquid_terminals, num_drybulk_terminals].
        terminal_data (pd.DataFrame): DataFrame containing terminal data.
        terminal_data_df (pd.DataFrame): DataFrame containing terminal data.
        terminal_tuple_cache (dict): Dictionary containing terminal tuple cache.
        seed (int): Random seed for reproducibility.
    Returns:
        None
    """
    initialize_rng(seed)

    train_data = pd.DataFrame(columns=[
        'train_id', 'terminal_id', 'cargo type', 'terminal', 
        'arrival at', 'car amount', 'cargo transfer rate', 
        'total transfer cargo', 'import', 'export'
    ])

    i = 0
    for terminal_type in ["Container", "Liquid", "DryBulk"]:
        for terminal_id in range(num_terminals[i]):
            terminal = terminal_id + 1
            interarrival_time = 1 / get_value_by_terminal(terminal_data, terminal_type, terminal, "train arrival rate")
            transfer_rate = get_value_by_terminal(terminal_data, terminal_type, terminal, "train cargo transfer rate")
            per_car_cargo = get_value_by_terminal(terminal_data, terminal_type, terminal, "train car payload size") 
            import_terminal = get_value_by_terminal(terminal_data, terminal_type, terminal, "import")
            export_terminal = get_value_by_terminal(terminal_data, terminal_type, terminal, "export")
            
            num_trains = int(SIMULATION_TIME / interarrival_time)

            if import_terminal and not export_terminal:
                import_bool = [True] * num_trains
                export_bool = [False] * num_trains
            elif not import_terminal and export_terminal:
                import_bool = [False] * num_trains
                export_bool = [True] * num_trains
            else:
                import_bool = []
                export_bool = []
                for _ in range(num_trains):
                    test = random.choice([True, False])
                    import_bool.append(test)
                    export_bool.append(not test)

            # Add subsequent trains
            car_amounts = get_values_by_terminal_random_sample(terminal_data_df, terminal_type, terminal, "train car amount", num_trains, seed)
            total_transfer_cargos = [per_car_cargo * car_amount for car_amount in car_amounts]
            arrivals = [j * interarrival_time for j in range(num_trains)]

            # Create the DataFrame directly
            subsequent_trains = pd.DataFrame({
                'train_id': [None] * num_trains,
                'terminal_id': [terminal_id] * num_trains,
                'terminal': [terminal] * num_trains,
                'cargo type': [terminal_type] * num_trains,
                'arrival at': arrivals,
                'car amount': car_amounts,
                'cargo transfer rate': [transfer_rate] * num_trains,
                'total transfer cargo': total_transfer_cargos,
                'import': import_bool,
                'export': export_bool
            })

            subsequent_trains = subsequent_trains.dropna(axis=1, how='all')
            subsequent_trains = subsequent_trains.dropna(axis=0, how='all')
            if not subsequent_trains.empty:
                train_data = train_data.dropna(axis=1, how='all')
                train_data = pd.concat([train_data, subsequent_trains], ignore_index=True)

        i += 1

    # Convert data types
    train_data['arrival at'] = pd.to_numeric(train_data['arrival at'], errors='coerce')
    train_data['terminal_id'] = train_data['terminal_id'].astype(int)
    train_data['terminal'] = train_data['terminal'].astype(int)
    train_data['car amount'] = train_data['car amount'].astype(int)
    train_data['cargo transfer rate'] = train_data['cargo transfer rate'].astype(float)
    train_data['total transfer cargo'] = train_data['total transfer cargo'].astype(float)
    train_data['import'] = train_data['import'].astype(bool)
    train_data['export'] = train_data['export'].astype(bool)
    train_data = train_data.sort_values('arrival at')
    train_data = train_data[train_data['arrival at'] != 0]

    # filter using TERMINALS_WITH_NO_TRAINS
    for cargo_type, list_of_terminals in TERMINALS_WITH_NO_TRAINS.items():
        for terminal in list_of_terminals:
            train_data = train_data[~((train_data['cargo type'] == cargo_type) & (train_data['terminal'] == terminal))]

    train_data.reset_index(drop=True, inplace=True)
    train_data['train_id'] = train_data.index + 1
    print("Number of trains", len(train_data))
    output_path = f'.{run_id}/logs/train_data.csv'
    train_data.to_csv(output_path, index=False)

def get_piplines_import(num_terminals_list, terminal_data):
    """
    Get the list of liquid terminals that have pipelines as source or sink.
    This function checks each liquid terminal to see if it has a pipeline source or sink.
    Args:
        num_terminals_list (tuple): A tuple containing the number of terminals for each type of ship (num_container_terminals, num_liquid_terminals, num_drybulk_terminals).
        terminal_data (pd.DataFrame): DataFrame containing terminal data.
    Returns:
        tuple: Two lists containing the liquid terminals with pipeline sources and sinks.
    """

    _, num_liquid_terminals, _ = num_terminals_list
    liq_terminals_with_pipeline_source = []
    liq_terminals_with_pipeline_sink = []

    for liquid_terminal in range(num_liquid_terminals):
        pipeline_source = get_value_by_terminal(terminal_data, "Liquid", liquid_terminal+1, "pipeline source")
        pipeline_sink = get_value_by_terminal(terminal_data, "Liquid", liquid_terminal+1, "pipeline sink")
        if pipeline_source:
            liq_terminals_with_pipeline_source.append(liquid_terminal+1)
        if pipeline_sink:
            liq_terminals_with_pipeline_sink.append(liquid_terminal+1)
    
    # print("Liquid terminals with pipeline source:", liq_terminals_with_pipeline_source)
    # print("Liquid terminals with pipeline sink:", liq_terminals_with_pipeline_sink)

    return liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink
        
