"""
This module provides functions to analyze the capacity of a queueing system,
calculate the ultimate capacity, and solve differential equations related to queue dynamics.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob


# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # Add parent directory to sys.path
# sys.path.append(parent_dir)

from constants import SCENARIO_NAME, NUM_MONTHS, ARRIVAL_INCREASE_FACTOR



def calculate_mean_wait_time(arrival_rates, queue_lengths, input_mean_wait_time, logfilename):
    """
    Calculate the mean wait time for a queueing system given arrival rates and queue lengths.

    Args:
        arrival_rates (list): List of 3 arrival rates for the queues.
        queue_lengths (list): List of 3 queue lengths for the queues.
        input_mean_wait_time (float): The input mean wait time to compare against.

    Returns:
        dict: A dictionary with calculated service rates, mean wait time, and relative error.
    """
    print("\nCapacity Analysis")
    print("Arrival Rates: ", arrival_rates)
    print("Queue Lengths: ", queue_lengths)
    with open(logfilename, 'a') as f:
        f.write("\nCapacity Analysis\n")
        f.write("Arrival Rates: " + str(arrival_rates) + "\n")
        f.write("Queue Lengths: " + str(queue_lengths) + "\n")
    arrival_rates = np.array(arrival_rates)
    queue_lengths = np.array(queue_lengths)
    eps = 1e-6

    # Define the fitting function
    def fitting_function(mu, arrival_rates, queue_lengths):
        numerator = np.sum(arrival_rates / mu**2)
        denominator = 1 - (arrival_rates / mu).sum()
        return (numerator / denominator) * arrival_rates - queue_lengths

    # Objective function for minimization
    def objective_function(mu, arrival_rates, queue_lengths):
        return np.sum(fitting_function(mu, arrival_rates, queue_lengths)**2)

    # Constraint to ensure the stability condition
    def stability_constraint(mu, arrival_rates):
        return 1 - (arrival_rates / mu).sum()

    # Initial guess for service rate
    initial_guess = [1.1 * float(arrival_rates.sum())]

    # Define constraints
    constraints = {
        'type': 'ineq',
        'fun': stability_constraint,
        'args': (arrival_rates,)
    }

    # Perform optimization
    result = minimize(
        objective_function,
        initial_guess,
        method='SLSQP',
        args=(arrival_rates, queue_lengths),
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000}
    )

    if result.success:
        mu = result.x[0]
        print(f"Calculated service rate (mu): {mu:.2f}")
        with open(logfilename, 'a') as f:
            f.write(f"Calculated service rate (mu): {mu:.2f}\n")
    else:
        raise ValueError("Optimization failed: " + result.message)

    # Calculate mean wait time
    numerator_W_q = np.sum(arrival_rates / mu**2)
    denominator_W_q = 1 - np.sum(arrival_rates / mu)
    calculated_mean_wait_time = numerator_W_q / denominator_W_q

    print(f"Calculated mean wait time: {calculated_mean_wait_time:.2f}")
    print(f"Input mean wait time: {input_mean_wait_time:.2f}")
    with open(logfilename, 'a') as f:
        f.write(
            f"Calculated mean wait time: {calculated_mean_wait_time:.2f}\n")
        f.write(f"Input mean wait time: {input_mean_wait_time:.2f}\n")

    # Calculate relative error
    relative_error = (calculated_mean_wait_time -
                      input_mean_wait_time) / input_mean_wait_time * 100
    print(f"Relative error (%): {relative_error:.2f}")

    # Return results as a dictionary
    output = {
        'Service Rate': mu,
        'Calculated Mean Wait Time': calculated_mean_wait_time,
        'Input Mean Wait Time': input_mean_wait_time,
        'Relative Error (%)': relative_error
    }

    print("Capacity Analysis Complete")
    print("Relative Error: ", relative_error)
    with open(logfilename, 'a') as f:
        f.write("Relative Error: " + str(relative_error) + "\n")
    return output


def dN_dlambda(lambda_, T, N, Cu, theta):
    """
    Differential equation for N with respect to lambda.
    Args:
        lambda_ (float): The current value of lambda.
        T (float): Total time in hours.
        N (float): Current value of N.
        Cu (float): Ultimate capacity.
        theta (float): Degree of congestion.
    Returns:
        float: The derivative of N with respect to lambda.
    """
    return T * (1 - (N / (T * Cu))**theta)


def solve_N(lambda_eval, Cu, T, theta):
    """
    Solve the ODE for N using the initial condition N(0) = 0.
    Args:
        lambda_eval (array): Array of lambda values for evaluation.
        Cu (float): Ultimate capacity.
        T (float): Total time in hours.
        theta (float): Degree of congestion.
    Returns:
        array: Solution for N at the given lambda values.
    """
    sol = solve_ivp(
        fun=lambda lam, N: dN_dlambda(lam, T, N, Cu, theta),
        t_span=(0, lambda_eval[-1]),
        y0=[0],
        t_eval=lambda_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-8
    )
    return sol.y[0]


def objective(params, constant_args):
    """
    Objective function to minimize the difference between predicted and actual N values.
    Args:
        params (list): List containing Cu and theta.
        constant_args (list): List containing a, b, c, d, and T.
    Returns:
        float: Sum of squared differences between predicted and actual N values.
    """
    Cu, theta = params
    a, b, c, d, T = constant_args
    lambda_vals = np.array([a, c])
    N_vals = np.array([b, d])
    N_pred = solve_N(lambda_vals, Cu, T, theta)
    return np.sum((N_pred - N_vals)**2)

def calculate_ultimate_capacity() -> tuple[float, float]:
    """
    Auto‐discover all scenario logs (BASE + others), but only read
    'Operating capacity:' from the BASE file. Fit dN/dλ = T*(1 - (N/(Cu*T))**θ),
    write results, and plot every sample labeled.
    """
    # 1) Setup
    T = NUM_MONTHS * 30 * 24
    LOG_DIR = 'bottleneckAnalysis/logs'

    # 2) Identify files
    base_file = os.path.join(LOG_DIR, f'{SCENARIO_NAME}_BASE.txt')
    pattern   = os.path.join(LOG_DIR, f'{SCENARIO_NAME}_*.txt')
    all_files = sorted(glob.glob(pattern))
    high_files = [f for f in all_files if f != base_file]

    # 3) Read BASE file
    lambdas = []
    Ns      = []
    Ns_actual = []
    op_capacity = None


    with open(base_file) as f:
        lam0 = N0 = None
        for line in f:
            if line.startswith('Operating capacity:'):
                op_capacity = float(line.split()[2])
            elif line.startswith('Arrival rate:'):
                lam0 = float(line.split()[2])
            elif line.startswith('Exited ships:'):
                N0 = float(line.split()[2])
        if lam0 is None or N0 is None:
            raise RuntimeError(f"Could not parse λ or N from BASE file {base_file}")
        lambdas.append(lam0)
        Ns.append(N0)
        Ns_actual.append(N0)

    # 4) Read every non-BASE file (only λ and N)
    for fn in high_files:
        lam = Ni = None
        with open(fn) as f:
            for line in f:
                if line.startswith('Arrival rate:'):
                    lam = float(line.split()[2])
                elif line.startswith('Exited ships:'):
                    Ni = float(line.split()[2])
        if lam is not None and Ni is not None:
            lambdas.append(lam)
            Ns.append(Ni)
            Ns_actual.append(Ni)
        else:
            print(f"Skipping {fn}: missing Arrival rate or Exited ships")

        
    lambdas = np.array(lambdas)
    Ns      = np.array(Ns)

    sort_idx = np.argsort(lambdas)
    lambdas = lambdas[sort_idx]
    Ns = Ns[sort_idx]

    print("Discovered sample points:")
    for i, (λ, N) in enumerate(zip(lambdas, Ns), start=1):
        print(f"  sample point {i}: λ = {λ:.2f},  N = {N:.2f}")

    # 5) Define multi‐point objective
    def multi_obj(x, lam_arr, N_arr, T):
        Cu, θ = x
        N_pred = solve_N(lam_arr, Cu, T, θ)
        return np.sum((N_pred - N_arr) ** 2)

    # 6) Optimize
    initial = [max(Ns)/T, 5.0]
    bounds  = [(0.1, max(lambdas)), (0.1, 20.0)] #TODO: adjust bounds if needed
    res = minimize(multi_obj, x0=initial, args=(lambdas, Ns, T), bounds=bounds)
    Cu_opt, theta_opt = res.x
    print(f"\nOptimal Cu = {Cu_opt:.4f},  θ = {theta_opt:.4f}")

    # 7) Append to results.txt
    out_path = 'bottleneckAnalysis/results.txt'
    with open(out_path, 'a') as f:
        f.write(f"===== Capacity Analysis for {SCENARIO_NAME} =====\n")
        f.write(f"Months: {NUM_MONTHS} (T = {T} hours)\n")
        f.write("Sample points (λ_i, N_i):\n")
        for i, (λ, N) in enumerate(zip(lambdas, Ns), start=1):
            f.write(f"  Point {i}: λ = {λ:.4f}, N = {N:.4f}\n")
        if op_capacity is not None:
            f.write(f"Operating capacity (Co): {op_capacity:.4f} vessels/hr\n")
        f.write(f"Fitted ultimate capacity (Cu): {Cu_opt:.4f}\n")
        f.write(f"Fitted exponent (θ): {theta_opt:.4f}\n\n")

    # 8) Plot everything
    λ_range = np.linspace(0, max(lambdas)*1.1, 300)
    N_fit   = solve_N(λ_range, Cu_opt, T, theta_opt)

    plt.figure(figsize=(6, 5))
    plt.plot(λ_range, N_fit, label='ODE fit', linewidth=2, color='orange')

    for i, (λ, N) in enumerate(zip(lambdas, Ns), start=1):
        if i == 1:
            plt.scatter(λ, N, s=20, label=f'Simulation runs', c='blue')
        else:
            plt.scatter(λ, N, s=20, c='blue')
        # plt.annotate(f"sample point {i}",
        #              (λ, N),
        #              textcoords="offset points",
        #              xytext=(5, 5),
        #              fontsize=9)

    plt.axhline(y=Cu_opt*T, linestyle='--', label='Ultimate capacity', color = 'magenta')
    # find y corresponding to operating capacity

    try:
        y = solve_N([op_capacity], Cu_opt, T, theta_opt) if op_capacity is not None else None
        plt.axvline(x=op_capacity, linestyle='--', label='Operating capacity', color ='green')
        plt.axhline(y=y, linestyle='--', color='green')

    except Exception as e:
        print(f"Error occurred while finding y for operating capacity: {e}")

    # set limits on y axis from 0 to 6500
    plt.ylim(0, 7000)
    plt.xlabel(r'$\lambda$', fontsize=14)
    plt.ylabel(r'$N^{exits}(\lambda)$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"bottleneckAnalysis/{SCENARIO_NAME}_capacity_analysis.pdf")
    plt.close()

    return Cu_opt, theta_opt
# calculate_ultimate_capacity()


