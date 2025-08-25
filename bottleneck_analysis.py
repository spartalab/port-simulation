import os
import shutil
import pandas as pd
import fileinput
import subprocess
import re
import sys
import ast

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(parent_dir)
from constants import NUM_PILOTS_DAY, NUM_PILOTS_NIGHT, NUM_TUGBOATS

# Helper to adjust a single cell (number or tuple-string)
def adjust_cell(cell, pct, col):
    """
    If `cell` is a single numeric value (or numeric string), scale it.
    If it's a tuple-string like "(1500, 2500)", parse, scale each entry,
    then return it in the same tuple format.
    """
    # 1) Try simple float conversion
    try:
        num = float(cell)
        if col == 'train arrival rate':
            return (num * (1 + pct))
        return int(round(num * (1 + pct)))
    except (ValueError, TypeError):
        pass

    # 2) Try parsing as a tuple or list literal
    try:
        tup = ast.literal_eval(cell)
        if isinstance(tup, (list, tuple)) and all(isinstance(v, (int, float)) for v in tup):
            scaled = tuple(int(round(v * (1 + pct))) for v in tup)
            # return with parentheses
            if isinstance(tup, tuple):
                return "(" + ", ".join(str(v) for v in scaled) + ")"
            else:
                return "[" + ", ".join(str(v) for v in scaled) + "]"
    except (ValueError, SyntaxError):
        pass

    # 3) Fallback: return as-is
    return cell

# Dictionary mapping scenario names to adjustment specs:
adjustments = {
    'Base': {'column': None, 'pct': 0.0},
    'LowCtrBerths': {'column': 'Berths', 'pct': -0.10, 'filter': {'Cargo': 'Container'}},
    'LowCtrStorage': {'column': 'storage volume', 'pct': -0.10, 'filter': {'Cargo': 'Container'}},
    'LowCtrTransfer': {'column': 'transfer rate per unit', 'pct': -0.10, 'filter': {'Cargo': 'Container'}},
    'LowLiqBerth': {'column': 'Berths', 'pct': -0.10, 'filter': {'Cargo': 'Liquid'}},
    'LowLiqStorage': {'column': 'storage volume', 'pct': -0.10, 'filter': {'Cargo': 'Liquid'}},
    'LowLiqTransfer': {'column': 'transfer rate per unit', 'pct': -0.10, 'filter': {'Cargo': 'Liquid'}},
    'LowDryBlkBerth': {'column': 'Berths', 'pct': -0.10, 'filter': {'Cargo': 'Dry Bulk'}},
    'LowDryBlkStorage': {'column': 'storage volume', 'pct': -0.10, 'filter': {'Cargo': 'Dry Bulk'}},
    'LowDryBlkeTransfer': {'column': 'transfer rate per unit', 'pct': -0.10, 'filter': {'Cargo': 'Dry Bulk'}},
    'SlowTruckArrival': {'column': 'truck arrival rate', 'pct': -0.10},
    'FastTruckArrival': {'column': 'truck arrival rate', 'pct':  0.10},
    'SlowTrainArrival': {'column': 'train arrival rate', 'pct': -0.10},
    'FastTrainArrival': {'column': 'train arrival rate', 'pct':  0.10},
    'LowPilots':        {'column': None,                     'pct': -0.10},
    'HighPilots':       {'column': None,                     'pct':  0.10},
    'LowTugs':          {'column': None,                     'pct': -0.10},
    'HighTugs':         {'column': None,                     'pct':  0.10},
}

# Arrival factor sets for double runs
arrival_factor_sets = [
    {'CTR': 6.0, 'LIQ': 6.0, 'DRYBULK': 6.0}, 
    {'CTR': 5.0, 'LIQ': 5.0, 'DRYBULK': 5.0},
    {'CTR': 4.0, 'LIQ': 4.0, 'DRYBULK': 4.0},
    {'CTR': 3.0, 'LIQ': 3.0, 'DRYBULK': 3.0},
    {'CTR': 2.0, 'LIQ': 2.0, 'DRYBULK': 2.0},
    {'CTR': 1.0, 'LIQ': 1.0, 'DRYBULK': 1.0},
]

def setup_scenario(scenario_name: str):
    cwd = os.getcwd()
    inputs_dir = os.path.join(cwd, 'inputs')
    base_csv    = os.path.join(inputs_dir, 'terminal_data_base.csv')
    target_csv  = os.path.join(inputs_dir, 'terminal_data.csv')
    const_file  = os.path.join(cwd, 'constants.py')
    backup    = os.path.join(cwd, 'config.py')


    # 1) Copy base CSV and constants.py
    shutil.copy(base_csv, target_csv)
    print(f"Copied {base_csv} -> {target_csv}")
    shutil.copy(backup, const_file)
    print(f"Copied {backup} -> {const_file}")

    # 2) Apply adjustment
    spec = adjustments.get(scenario_name)
    pct  = spec['pct']
    if spec is None or spec['column'] is None:
        print(f"No adjustment for '{scenario_name}', using base CSV.")
    else:
        col  = spec['column']
        filt = spec.get('filter')

        df = pd.read_csv(target_csv, dtype=str)    # read everything as str
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {target_csv}")

        if filt:
            # build mask
            mask = pd.Series(True, index=df.index)
            desc = []
            for k, v in filt.items():
                if k not in df.columns:
                    raise KeyError(f"Filter column '{k}' not found in {target_csv}")
                mask &= df[k] == v
                desc.append(f"{k}={v}")

            df.loc[mask, col] = df.loc[mask, col].apply(lambda x: adjust_cell(x, pct, col))
            print(f"Applied {pct*100:+.0f}% to '{col}' for rows where {', '.join(desc)}")
        else:
            df[col] = df[col].apply(lambda x: adjust_cell(x, pct, col))
            print(f"Applied {pct*100:+.0f}% to entire column '{col}'")

        df.to_csv(target_csv, index=False)

    # 3) Update SCENARIO_NAME in constants.py
    _update_constant(const_file, 'SCENARIO_NAME', f'"{scenario_name}"')
    print(f"Set SCENARIO_NAME = '{scenario_name}' in constants.py")

    # Update pilots or tugs if applicable
    if scenario_name in ('LowPilots', 'HighPilots') or scenario_name in ('LowTugs', 'HighTugs'):
        # use pct from adjustments
        if scenario_name.endswith('Pilots'):
            # scale NUM_PILOTS_DAY and NUM_PILOTS_NIGHT
            new_day   = tuple(int(round(x * (1 + pct))) for x in NUM_PILOTS_DAY)
            new_night = tuple(int(round(x * (1 + pct))) for x in NUM_PILOTS_NIGHT)
            _update_specific_constant(const_file, 'NUM_PILOTS_DAY',   f"({new_day[0]}, {new_day[1]})")
            _update_specific_constant(const_file, 'NUM_PILOTS_NIGHT', f"({new_night[0]}, {new_night[1]})")
            print(f"Set pilots for {scenario_name} ({pct*100:+.0f}%) in constants.py")
        else:
            # scale NUM_TUGBOATS
            new_tugs = tuple(int(round(x * (1 + pct))) for x in NUM_TUGBOATS)
            _update_specific_constant(const_file, 'NUM_TUGBOATS', f"({new_tugs[0]}, {new_tugs[1]})")
            print(f"Set tugs for {scenario_name} ({pct*100:+.0f}%) in constants.py")

    # 4) Run main.py twice with different arrival factors
    for factors in arrival_factor_sets:
        _update_specific_constant(const_file, 'ARRIVAL_INCREASE_FACTOR_CTR',    factors['CTR'])
        _update_specific_constant(const_file, 'ARRIVAL_INCREASE_FACTOR_LIQ',    factors['LIQ'])
        _update_specific_constant(const_file, 'ARRIVAL_INCREASE_FACTOR_DRYBULK',factors['DRYBULK'])
        print(f"Running main.py with factors CTR={factors['CTR']}, LIQ={factors['LIQ']}, DRYBULK={factors['DRYBULK']}")
        try:
            subprocess.run(['python', 'main.py'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"  → main.py failed: {e}")

    # 5) Restore original constants.py from backup
    shutil.copy(backup, const_file)
    print(f"Re-restored constants.py from {backup}")


def _update_constant(filepath: str, var_name: str, new_value):
    count = 0
    for line in fileinput.input(filepath, inplace=True):
        if line.strip().startswith(var_name) and count == 0:
            print(f"{var_name} = {new_value}")
            count += 1
        else:
            print(line, end='')

def _update_specific_constant(filepath: str, var_name: str, new_value):
    pattern = re.compile(rf'^\s*{re.escape(var_name)}\s*=')
    for line in fileinput.input(filepath, inplace=True):
        if pattern.match(line):
            print(f"{var_name} = {new_value}")
        else:
            print(line, end='')

def parse_capacity(path, csv_path=None):
    """
    Read a capacity-analysis text file, extract Operating Capacity (Co) and Fitted Ultimate Capacity (Cu) for each scenario,
    compute percent change in Cu relative to the Base scenario, and return a pandas DataFrame.
    If csv_path is provided, also save the DataFrame to CSV.

    Parameters:
    - path (str): path to the input text file.
    - csv_path (str, optional): path to write CSV output.

    Returns:
    - pd.DataFrame with columns ['Scenario', 'Co', 'Cu', 'Delta_Cu_Pct'].
    """
    # Read file
    with open(path, 'r') as f:
        text = f.read()

    # Find each scenario block
    pattern = re.compile(
        r"===== Capacity Analysis for\s+(?P<scenario>.+?)\s+=====\s*(?P<block>.*?)(?=(?:===== Capacity Analysis for)|\Z)",
        re.DOTALL
    )
    entries = []
    base_cu = None

    # Parse Co and Cu
    for m in pattern.finditer(text):
        name = m.group('scenario').strip()
        block = m.group('block')
        co_m = re.search(r"Operating capacity \(Co\):\s*([0-9]+\.?[0-9]*)", block)
        cu_m = re.search(r"Fitted ultimate capacity \(Cu\):\s*([0-9]+\.?[0-9]*)", block)
        if not (co_m and cu_m):
            continue
        co = float(co_m.group(1))
        cu = float(cu_m.group(1))
        entries.append({'Scenario': name, 'Co': co, 'Cu': cu})
        if name.lower() == 'base':
            base_cu = cu

    if base_cu is None:
        raise ValueError("Base scenario not found in input text.")

    # Build DataFrame and compute percent change
    df = pd.DataFrame(entries)
    df['Delta_Cu_Pct'] = (df['Cu'] - base_cu) / base_cu * 100

    # Save CSV if requested
    if csv_path:
        df.to_csv(csv_path, index=False)

    return df

if __name__ == '__main__':
    # scenario_name_list = [ 'Base',
    #                        'LowLiqBerth', 'SlowTruckArrival', 'LowPilots', 'LowLiqStorage', 'LowLiqTransfer',
    #                         'LowDryBlkBerth', 'LowDryBlkStorage', 'LowDryBlkeTransfer',
    #                       'LowCtrBerths', 'LowCtrStorage', 'LowCtrTransfer',
    #                        'FastTruckArrival', 'FastTrainArrival', 'SlowTrainArrival',
    #   'HighPilots','LowTugs','HighTugs']
    scenario_name_list = ['LowLiqBerth']
    
    # clear results from earlier runs in bottleneckAnalysis/results.txt
    with open("bottleneckAnalysis/results.txt", "w") as f:
        f.write("")

    for scenario in scenario_name_list:
        try:
            setup_scenario(scenario)
        except Exception as e:
            print(f"Error setting up scenario '{scenario}': {e}")
    print("All scenarios processed.")

    result_df = parse_capacity("bottleneckAnalysis/results.txt", "bottleneckAnalysis/capacity_analysis.csv")
    print(result_df)


#  