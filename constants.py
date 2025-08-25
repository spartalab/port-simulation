import os
import pandas as pd

# set directory
directory = os.getcwd()  

# set simulation IDs and run characteristics
WARMUP_ITERS = 1000  # Number of ships to ignore while calculating means, recommended: first 1,000
NUM_MONTHS = 4  # Number of months to run the simulation
SIMULATION_TIME = int(NUM_MONTHS*30*24) # Total time to run the simulation for, in hours

LOG_NUM = 10
NUM_RUNS = 2  # How many unique replications of different seeds to conduct for a given run
NUM_CORES = 2  # How many CPU cores to run individual seeds on simultaneously (set depending on local hardware capabilities)
START_SEED = 100  # Initial random seed for the simulation

# Cleans output result folders
DELETE_EVERYTHING = False # Cleans all existing result folders, including collated results (SAVE ALL DESIRED DATA PRIOR TO RUNNING AS TRUE)
DELETE_RESULTS_FOLDER = True  # Cleans the results folder only

SCENARIO_NAME = "LowLiqBerth"

# Set "what-if" scenarios. Set all as "False" for base conditions
MODEL_HURRICANE = False
MODEL_FOG = False
EXPANDED_CHANNEL = False

# Set truck and pipeline overrides to ensure terminal storage capacities remain
CTR_TRUCK_OVERRIDE = False # Set False if running bottleneck analysis; True for stable yard queue
LIQ_PIPELINE_OVERRIDE = False # Set False if running bottleneck analysis; True for stable yard queue

########################################################################################
# The following variables have been calibrated for a simulation of the a random port   #
# Change with ACTUAL port data.                                                        #
########################################################################################

if MODEL_FOG == True:
    INBOUND_CLOSED = [(1500, 1520), (1524, 1540), (1550, 1586), (1594, 1608), (1630, 1658), (1664, 1695), (1705, 1733), (
        1744, 1764), (1932, 1972), (1981, 1999), (2007, 2028), (2038, 2050), (2062, 2078), (2086, 2118), (2130, 2178), (2188, 2200)]
    OUTBOUND_CLOSED = [(1500, 1515), (1539, 1547), (1567, 1579), (1602, 1615), (1629, 1650), (1673, 1695), (1709, 1725), (
        1739, 1750), (2002, 2023), (2033, 2049), (2065, 2078), (2100, 2117), (2137, 2161), (2184, 2206), (2226, 2235), (2257, 2270)]
    FOG_CLOSURES = []  # BOTH INBOUND AND OUTBOUND CLOSURES
else:
    INBOUND_CLOSED = []
    OUTBOUND_CLOSED = []
    FOG_CLOSURES = []

# set as 1.0 for most cases, values change directly if running breakpoint analysis
ARRIVAL_INCREASE_FACTOR_CTR = 6.0
ARRIVAL_INCREASE_FACTOR_LIQ = 6.0
ARRIVAL_INCREASE_FACTOR_DRYBULK = 6.0

ARRIVAL_INCREASE_FACTOR = ARRIVAL_INCREASE_FACTOR_CTR

if ARRIVAL_INCREASE_FACTOR != 1.0:
    print(f"\n\nWARNING!: Arrival increase factor set to {ARRIVAL_INCREASE_FACTOR} for all cargo types.")

ARRIVAL_INCREASE_FACTOR = round(ARRIVAL_INCREASE_FACTOR, 2)

# percent of total vessel tonnage accounted for as vessel weight
NON_CARGO_DEAD_WEIGHT_PERCENT_CTR = 0.2
NON_CARGO_DEAD_WEIGHT_PERCENT_DK = 0.1
NON_CARGO_DEAD_WEIGHT_PERCENT_LIQ = 0.1

# anchorage waiting times calibrated to each cargo type 
# These are unexplained wait times (additional waiting times account for factors not modelled
# in simulation explicitely. First run with 0 values then caliberate based on observed
# data and simulation outputs. 
ANCHORAGE_WAITING_CONTAINER = 0
ANCHORAGE_WAITING_LIQUID = 0
ANCHORAGE_WAITING_DRYBULK = 0

# terminal efficiency times, used to extend the vessel time in port for each cargo type
# These are efficiency factors for each terminal. Should be between 0 and 1
# A Value of one means terminal is fully efficient and no delays.
# First run with 0 values then caliberate based on observed
# data and simulation outputs. 
CTR_TERMINAL_EFFICIENCY = 1.0
LIQ_TERMINAL_EFFICIENCY = 1.0
DRYBULK_TERMINAL_EFFICIENCY = 1.0

# removes combined beam and draft restrictions if set to "True"
CHANNEL_SAFETWOWAY = False

# set number of pilots and tugs
# Dummy values, set as actual port values
NUM_PILOTS_DAY = (50, 60)
NUM_PILOTS_NIGHT = (20, 30)
NUM_TUGBOATS = (50, 100)

# set channel entrance and navigation times
TIME_COMMON_CHANNEL = (2, 3)  # in hours 
TIME_FOR_TUG_STEER = (0.2, 0.4)  # in hours
TIME_FOR_UTURN = (0.2, 0.4)  # in hours

# set min and max range for arriving truck wait times, in hours
TRUCK_WAITING_TIME = (1/60, 5/60)

# interarrival times derived from AIS data.
# Look for Bathgate et al. (2026) for procedure to obtain these values.
# Currently values set as dummy values. (All values in hours)
mean_interarrival_time_container = 10 * 1/ARRIVAL_INCREASE_FACTOR_CTR
mean_interarrival_time_gencargo = 5 * 1/ARRIVAL_INCREASE_FACTOR_DRYBULK
mean_interarrival_time_tanker = 1 * 1/ARRIVAL_INCREASE_FACTOR_LIQ
mean_interarrival_time_total = mean_interarrival_time_container + mean_interarrival_time_gencargo + mean_interarrival_time_tanker
base_arrival_rate = 1/10 + 1/5 + 1/1

# set truncation values for 
max_interaarival_ctr = 100000000000000000
max_interaarival_liq = 100000000000000000
max_interaarival_drybulk = 100000000000000000

min_interarrival_liquid = 0.0001

# Set terminals with no truck or rail connections. 
# Note: These overrides terminal data; Example shown below:
# Example: TERMINALS_WITH_NO_TRUCKS = {"Liquid": [1]}
TERMINALS_WITH_NO_TRUCKS = {"Liquid": [], "DryBulk": [], "Liquid": []}
TERMINALS_WITH_NO_TRAINS = {"Container": [], "Liquid": [], "DryBulk": []}  

# Rate for landside pipelines connection (these act as sorce or sink)
# Note this is different from pipelines that connects to vessels.
PIPELINE_RATE = 100  # cbm/timestep

# Model the channel
NUM_CHANNEL_SECTIONS = 10

# Model portion after which wider ships are not allowed.
# Helps in modelling channels that become narrower. 
# certain channel segments might only accessible to smaller vessels, list the start terminal for these conditions here
# Set as large number if not applicable
NO_LARGE_SHIP_BEYOND_THIS_TERMINAL_CTR = 2
NO_LARGE_SHIP_BEYOND_THIS_TERMINAL_LIQ = 10
NO_LARGE_SHIP_BEYOND_THIS_TERMINAL_DRYBULK = 10

# Set what beam determines when wider ships are not allowed 
MAX_BEAM_SMALL_SHIP = 106

# common liquid bulk cargo density conversion factors for more accurate cargo payload estimation
LIQUID_CONVERSION_FACTORS = [
    1.0,    # Water (1 ton = 1 CBM)
    1.18,   # Crude oil (1 ton = 1.18 CBM)
    1.16,   # Diesel (1 ton = 1.16 CBM)
    1.38,   # Gasoline (1 ton = 1.38 CBM)
    0.92,   # Liquid ammonia (1 ton = 0.92 CBM)
    0.78,   # Molasses (1 ton = 0.78 CBM)
    0.87,   # Vegetable oil (1 ton = 0.87 CBM)
    1.11,   # Ethanol (1 ton = 1.11 CBM)
    1.03,   # Jet fuel (1 ton = 1.03 CBM)
    1.03,   # Kerosene (1 ton = 1.03 CBM)
    1.22,   # LPG (1 ton = 1.22 CBM)
    1.25    # LNG (1 ton = 1.25 CBM)
]

CONTAINER_CONVERSION_FACTORS = [
    1/24,     # 20' container can carry 24 tons
    1/33      # 40' container can carry 33 tons
]

# import terminal data from CSV in "inputs" folder
df = pd.read_csv('./inputs/terminal_data.csv')

# set daylight restriction hours
# daylight hours are 7 AM till 7 PM (19:00)
# hours for daylight restriction due to expanded channel are 4 AM till 9 PM (21:00)
if EXPANDED_CHANNEL == True:
    START_DAYLIGHT_RESTRICTION_HR = 4
    STOP_DAYLIGHT_RESTRICTION_HR = 21
else:
    START_DAYLIGHT_RESTRICTION_HR = 7
    STOP_DAYLIGHT_RESTRICTION_HR = 19

# set channel dimensions (current or expanded)
if EXPANDED_CHANNEL == True:
    channel_dimensions = pd.read_csv(
        './inputs/channel_dimensions_expanded.csv')
else:
    channel_dimensions = pd.read_csv('./inputs/channel_dimensions.csv')

# read channel data from CSV in "inputs" folder
dict_from_csv = channel_dimensions.to_dict(orient="index")
CHANNEL_SECTION_DIMENSIONS = {list(value.values())[0]: tuple(
    list(value.values())[1:]) for key, value in dict_from_csv.items()}
LAST_SECTION_DICT = {cargo: dict(
    zip(group['Terminal'], group['Segment'])) for cargo, group in df.groupby('Cargo')}
BERTHS_CTR_TERMINAL = df[df['Cargo'] == 'Container']['Berths'].tolist()
BERTHS_LIQ_TERMINAL = df[df['Cargo'] == 'Liquid']['Berths'].tolist()
BERTH_DRYBULK_TERMINAL = df[df['Cargo'] == 'DryBulk']['Berths'].tolist()

# conversion factors
TIME_FACTOR = 60  # 1 hour = 60 minutes (Channel timestep is in minutes)

# in hours, minimum spacing between entering inbound vessels (5/60 is 5 minutes)
CHANNEL_ENTRY_SPACING = 5/60
