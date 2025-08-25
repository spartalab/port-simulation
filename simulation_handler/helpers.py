""" 
Helper functions for the simulation environment.
This module contains various utility functions for logging, data cleaning,
and random number generation used in the simulation.
"""

import os
import shutil
import re
import random
import os
import glob

import numpy as np
from scipy.stats import truncnorm
import pandas as pd

import constants

#######################
# Logs
#######################


def log_line(run_id, filename, line):
    """
    Logs a line to a specified file in the run's logs directory.
    Args:
        run_id (str): Unique identifier for the run to save the log.
        filename (str): Name of the file to log the line.
        line (str): The line to log.
    Returns:
        None: Appends the line to the specified log file.
    """
    try:
        with open(f'.{run_id}/logs/{filename}', 'a') as f:
            f.write(line + '\n')
    except FileNotFoundError:
        with open(f'.{run_id}/logs/{filename}', 'w') as f:
            f.write(line + '\n')


def save_warning(run_id, message, print_val=True):
    """
    Save a warning message to the logs directory for the current run.
    If the logs directory does not exist, it will be created.
    Args:
        run_id (str): The identifier for the current run.
        message (str): The warning message to be saved.
        print_val (bool): Whether to print the message to the console. Default is True.
    Returns:
        None
    """
    if not os.path.exists(f'.{run_id}/logs/warnings.txt'):
        with open(f'.{run_id}/logs/warnings.txt', 'w') as f:
            f.write('Warnings:\n')
        with open(f'.{run_id}/logs/warnings.txt', 'a') as f:
            f.write(f'{message}\n')
    else:
        with open(f'.{run_id}/logs/warnings.txt', 'a') as f:
            f.write(f'{message}\n')
    if print_val:
        print(f"{message}")


def clear_env(env, ship_proc, truck_proc, train_proc, data_taker_proc):
    """
    Clear the environment and interrupt all processes.
    Args:
        env (Simpy.Environment): The simulation environment to be cleared.
        ship_proc (Simpy.Process): The process for handling ships.
        truck_proc (Simpy.Process): The process for handling trucks.
        train_proc (Simpy.Process): The process for handling trains.
        data_taker_proc: The process for taking data.
    Returns:
        None
    """
    try:
        ship_proc.interrupt()
    except:
        pass
    try:
        truck_proc.interrupt()
    except:
        pass
    try:
        train_proc.interrupt()
    except:
        pass

    try:
        data_taker_proc.interrupt()
    except:
        pass

    env._queue.clear()


def clear_logs(run_id):
    """
    Clear the logs directory for the current run by removing all files and subdirectories.
    Then create the necessary subdirectories again.
    This function is useful for resetting the logs directory before starting a new simulation run.
    Args:
        run_id (str): The identifier for the current run.
    Returns:
        None
    """

    if constants.DELETE_RESULTS_FOLDER == False:
        shutil.rmtree(f'.{run_id}/logs', ignore_errors=True)
        shutil.rmtree(f'.{run_id}/plots', ignore_errors=True)
        # shutil.rmtree(f'.{run_id}/animations', ignore_errors=True)
        shutil.rmtree(f'.{run_id}/bottlenecks', ignore_errors=True)

    # os.makedirs(f'.{run_id}/animations')

    os.makedirs(f'.{run_id}/logs')
    os.makedirs(f'.{run_id}/logs/availability')
    os.makedirs(f'.{run_id}/plots')
    os.makedirs(f'.{run_id}/bottlenecks')
    os.makedirs(f'.{run_id}/bottlenecks/chassisBays/')
    os.makedirs(f'.{run_id}/plots/TerminalProcessCharts/')
    os.makedirs(f'.{run_id}/plots/TerminalProcessDist/')
    os.makedirs(f'.{run_id}/plots/DwellTimes/')
    os.makedirs(f'.{run_id}/plots/TurnTimes/')
    os.makedirs(f'.{run_id}/plots/DwellTimesDist/')
    os.makedirs(f'.{run_id}/plots/TurnTimesDist/')
    os.makedirs(f'.{run_id}/plots/scenarios')
    os.makedirs(f'.{run_id}/plots/shipDistribution')

    os.makedirs(f'.{run_id}/plots/truckDistribution')
    os.makedirs(f'.{run_id}/plots/TruckDwellByCargo')
    os.makedirs(f'.{run_id}/plots/TruckDwellByTerminal')
    os.makedirs(f'.{run_id}/plots/TruckArrivalByCargo')
    os.makedirs(f'.{run_id}/plots/TruckArrivalByTerminal')

    # Create a force_action.txt file under logs
    with open(f'.{run_id}/logs/force_action.txt', 'w') as f:
        f.write('Force Actions')
        f.write('\n')


def clean_results_directory():
    """
    Remove existing result files and directories matching the pattern `.Results*`.
    This function is useful for cleaning up the results directory before starting a new simulation run.
    Args:
        None
    Returns:
        None
"""
    for file_path in glob.glob(".Results*"):
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                for root, dirs, files in os.walk(file_path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(file_path)
        except OSError as e:
            print(f"Error removing {file_path}: {e}")

#######################
# Data cleaning
#######################


def initialize_rng(seed):
    """
    Initialize random state objects for repeatable randomness across runs.
    Args:
        seed (int): The seed value for the random number generator.
    Returns:
        None
    """
    global rng_random, rng_numpy
    rng_random = random.Random(seed)
    rng_numpy = np.random.default_rng(seed)


def clean_data(directory):
    """
    Clean the terminal data by parsing ranges and converting them to tuples.
    This function reads a CSV file containing terminal data, processes the range strings,
    and converts them into tuples of integers or floats.
    Args:
        directory (str): The directory where the terminal data CSV file is located.
    Returns:
        pd.DataFrame: A DataFrame containing the cleaned terminal data with ranges parsed into tuples.
    """

    def parse_range(range_str):
        """
        Parse a range in the format '(a,b)' from a string into a tuple of integers or floats.
        If the input is a single number, it will be returned as an integer.
        Args:
            range_str (str): The range string to be parsed.
        Returns:
            tuple or int: A tuple of integers if the input is a range, or an integer if it's a single number.
        """
        # convert range string to a string
        range_str = str(range_str)
        try:
            match = re.match(r"\(([\d.]+)\s*,\s*([\d.]+)\)", range_str)
            start, end = int(match.group(1)), int(match.group(2))
            return (start, end)
        except:
            return int(range_str)

    csv_data = pd.read_csv(f'{directory}/inputs/terminal_data.csv')
    for column in csv_data.columns[3:]:
        csv_data[column] = csv_data[column].apply(
            lambda x: parse_range(x) if isinstance(x, str) else x)
    return csv_data


def create_terminal_data_cache(terminal_data, run_id, seed):
    """
    Create a cache dictionary from terminal data for quick access to resource values.
    This function initializes a random number generator with a given seed,
    iterates through the terminal data, and populates a cache dictionary with keys
    based on cargo type, terminal ID, and resource name. If the resource value is a range,
    it stores a random integer from that range; otherwise, it stores the value directly.
    Args:
        terminal_data (pd.DataFrame): The DataFrame containing terminal data.
        run_id (str): The identifier for the current run, used for logging.
        seed (int): The seed value for the random number generator.
    Returns:
        dict: A dictionary where keys are tuples of (cargo_type, terminal_id, resource_name)
              and values are the corresponding resource values or random integers from ranges.
    """
    # Create the cache dictionary
    initialize_rng(seed)
    cache = {}
    # Populate the cache
    for index, row in terminal_data.iterrows():
        cargo_type = row['Cargo']
        terminal_id = row['Terminal']
        # Skip 'Cargo' and 'Terminal' columns
        for resource_name in row.index[2:]:
            value = row[resource_name]
            key = (cargo_type, terminal_id, resource_name)
            # If value is a tuple-like string, convert it to tuple and store a random range value; otherwise, store as int
            if isinstance(value, tuple):
                # Convert to tuple and store a random value from range
                cache[key] = rng_random.randint(value[0], value[1])
            else:
                if resource_name in ['train arrival rate', 'truck arrival rate']:
                    cache[key] = value
                # Store integer or directly as it is
                else:
                    cache[key] = int(value)

    return cache


def get_value_by_terminal(terminal_data_cache, cargo_type, terminal_id, resource_name):
    """
    Retrieve a value from the terminal data cache based on cargo type, terminal ID, and resource name.
    This function checks the cache for a specific key and returns the corresponding value.
    Args:
        terminal_data_cache (dict): The cache dictionary containing terminal data.
        cargo_type (str): The type of cargo.
        terminal_id (str): The ID of the terminal.
        resource_name (str): The name of the resource.
    Returns:        
        The value associated with the specified cargo type, terminal ID, and resource name,
        or None if the key does not exist in the cache.
    """
    key = (cargo_type, terminal_id, resource_name)
    return terminal_data_cache.get(key)


def create_terminal_tuple_cache(terminal_data, run_id, seed):
    """
    Create a cache dictionary from terminal data for quick access to resource values as tuples.
    This function initializes a random number generator with a given seed,
    iterates through the terminal data, and populates a cache dictionary with keys
    based on cargo type, terminal ID, and resource name. If the resource value is a range,
    it stores the range as a tuple; otherwise, it stores the value as a tuple of itself.
    Args:
        terminal_data (pd.DataFrame): The DataFrame containing terminal data.   
        run_id (str): The identifier for the current run, used for logging.
        seed (int): The seed value for the random number generator.
    Returns:
        dict: A dictionary where keys are tuples of (cargo_type, terminal_id, resource_name)
              and values are tuples of the corresponding resource values or ranges.
    """
    # Create the cache dictionary
    initialize_rng(seed)
    truck_load_cache = {}

    # Populate the cache
    for index, row in terminal_data.iterrows():
        cargo_type = row['Cargo']
        terminal_id = row['Terminal']
        # Skip 'Cargo' and 'Terminal' columns
        for resource_name in row.index[2:]:
            value = row[resource_name]
            key = (cargo_type, terminal_id, resource_name)
            # If value is a tuple-like string, convert it to tuple and store a random range value; otherwise, store as int
            if isinstance(value, tuple):
                # Convert to tuple and store a random value from range
                truck_load_cache[key] = (value[0], value[1])
            else:
                if resource_name in ['train arrival rate', 'truck arrival rate']:
                    truck_load_cache[key] = (value, value)
                # Store integer or directly as it is
                else:
                    truck_load_cache[key] = (int(value), int(value))

    return truck_load_cache


def get_value_from_terminal_tuple(terminal_tuple_cache, cargo_type, terminal_id, resource_name):
    """
    Retrieve a value from the terminal tuple cache based on cargo type, terminal ID, and resource name.
    This function checks the cache for a specific key and returns a random integer from the range stored in the tuple.
    Args:
        terminal_tuple_cache (dict): The cache dictionary containing terminal data as tuples.
        cargo_type (str): The type of cargo.        
        terminal_id (str): The ID of the terminal.
        resource_name (str): The name of the resource.
    Returns:
        int: A random integer from the range stored in the tuple for the specified cargo type, terminal ID, and resource name.
        If the key does not exist in the cache, it returns None.
    """
    # initialize_rng(seed)
    key = (cargo_type, terminal_id, resource_name)
    tups = terminal_tuple_cache.get(key)
    return rng_random.randint(tups[0], tups[1])


def get_values_by_terminal_random_sample(terminal_data, cargo_type, terminal_id, resource_name_input, num_samples, seed):
    """
    Get a list of random samples from the terminal data based on cargo type and terminal ID.
    This function retrieves a specific resource value from the terminal data for a given cargo type and terminal ID,
    and generates a list of random samples from that value. If the value is a range, it samples from that range;
    otherwise, it returns a list with the value repeated for the specified number of samples.  
    Args:
        terminal_data (pd.DataFrame): The DataFrame containing terminal data.
        cargo_type (str): The type of cargo.
        terminal_id (str): The ID of the terminal.
        resource_name_input (str): The name of the resource to sample from.
        num_samples (int): The number of random samples to generate.
        seed (int): The seed value for the random number generator.
    Returns:
        list: A list of random samples from the specified resource value.
        If the resource value is a range, it returns random integers from that range;
        otherwise, it returns a list with the value repeated for the specified number of samples.
    """
    random.seed(seed)

    row = terminal_data[(terminal_data['Terminal'] == terminal_id) &
                        (terminal_data['Cargo'] == cargo_type)].iloc[0]

    value = row[resource_name_input]

    try:
        return [random.randint(int(value[0]), int(value[1])) for _ in range(num_samples)]
    except:
        return [value] * num_samples


#######################
# Helper functions
#######################

def is_daytime(time, start=6, end=18):
    """
    Check if the given time is within the daytime range.
    Args:
        time (int or float): The time to check, in hours (0-23).
        start (int): The start of the daytime range (default is 6).
        end (int): The end of the daytime range (default is 18).
    Returns:
        bool: True if the time is within the daytime range, False otherwise.
    """
    return start <= time % 24 <= end

#######################
# Random
#######################


def normal_random_with_sd(mu, sigma, seed, scale_factor=1):
    """
    Generate a random value from a truncated normal distribution with specified mean and standard deviation.
    The distribution is truncated to ensure the value is within a specified range based on the mean and standard deviation.
    Args:
        mu (float): The mean of the normal distribution.
        sigma (float): The standard deviation of the normal distribution.
        seed (int): Random seed for reproducibility.
        scale_factor (float): Factor to scale the standard deviation for truncation (default is 1).
    Returns:
        float: A random value from the truncated normal distribution, ensuring it is within the range [a, b].
    """
    np.random.seed(seed)
    a = max(0, mu - scale_factor * sigma)
    b = max(0, mu + scale_factor * sigma)
    if a <= 0:
        a = 0
    lower, upper = (a - mu) / sigma, (b - mu) / sigma
    distribution = truncnorm(lower, upper, loc=mu, scale=sigma)
    val = distribution.rvs()
    if val < a:
        return a
    elif val > b:
        return b
    return distribution.rvs()


def normal_random_with_limit(a, b, seed):
    """
    Generate a random value from a truncated normal distribution between a and b.
    Args:
        a (float): Lower limit of the range.
        b (float): Upper limit of the range.
        seed (int): Random seed for reproducibility.
    Returns:
        float: A random value from the truncated normal distribution.
    """
    np.random.seed(seed)
    mean = (a + b) / 2
    std_dev = (b - a) / 6
    lower_bound = (a - mean) / std_dev
    upper_bound = (b - mean) / std_dev
    distribution = truncnorm(lower_bound, upper_bound, loc=mean, scale=std_dev)
    return distribution.rvs()
