"""
Creates class object for Trucks.
"""
import constants
import random
from simulation_handler.helpers import get_value_by_terminal, log_line
from simulation_classes.pipeline import Pipeline

# Initialize the constants from the constants file
TRUCK_WAITING_TIME_MIN, TRUCK_WAITING_TIME_MAX = constants.TRUCK_WAITING_TIME
actions = {
    (1, 1): "both",
    (1, 0): "load",
    (0, 1): "unload",
    (0, 0): None
}


class Truck(object):
    """
    Truck class to simulate the loading and unloading of trucks at different terminals.
    Args:
        env (simpy.Environment): Simulation environment.
        liq_terminals_with_pipeline_source (list): List of liquid terminals with pipeline source.
        liq_terminals_with_pipeline_sink (list): List of liquid terminals with pipeline sink.       
        chassis_bays_utilization (dict): Dictionary to track chassis bays utilization.
        truck_id (int): Unique identifier for the truck.
        run_id (int): Unique identifier for the simulation run.
        terminal_type (str): Type of the terminal (e.g., "Container", "Liquid", "DryBulk").
        terminal_id (int): Unique identifier for the terminal, starts from 1.
        container_amount (tuple): Amount of containers to load and unload.
        liquid_amount (int): Amount of liquid to load or unload.
        drybulk_amount (int): Amount of dry bulk to load or unload.
        loading_bays (simpy.Resource): Resource representing the loading bays.
        port_tanks (simpy.Container): Container representing the liquid storage tanks at the port.
        truck_chassis (simpy.Resource): Resource representing the truck chassis.
        port_yard (simpy.Store): Store representing the container yard at the port.
        port_silos (simpy.Container): Container representing the dry bulk storage silos at the port.
        drybulk_bays (simpy.Resource): Resource representing the dry bulk loading bays.
        events (list): List to store events for logging.
        seed (int): Random seed for reproducibility.
    """

    def __init__(self, env, liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink, chassis_bays_utilization, truck_id, run_id, terminal_type, terminal_id, container_amount, liquid_amount, drybulk_amount, loading_bays, port_tanks, truck_chassis, port_yard, port_silos, drybulk_bays, events, seed, terminal_data):

        self.env = env
        self.events = events
        self.truck_id = truck_id
        self.terminal_type = terminal_type
        self.terminal_id = terminal_id  # starts from 1
        self.terminal_data = terminal_data
        self.run_id = run_id
        self.chassis_bays_utilization = chassis_bays_utilization

        self.container_amount = container_amount
        self.liquid_amount = liquid_amount
        self.drybulk_amount = drybulk_amount

        self.transfer_rate = get_value_by_terminal(
            self.terminal_data, terminal_type, terminal_id, 'truck loading/unloading rate')

        self.loading_bays = loading_bays
        # port_tanks_liquid_terminals[terminal_id]
        self.port_tanks = port_tanks

        self.truck_chassis = truck_chassis
        # port_yard_container_terminals[terminal_id]
        self.port_yard = port_yard

        # port_silos_drybulk_terminals[terminal_id]
        self.port_silos = port_silos
        self.drybulk_bays = drybulk_bays

        self.liq_terminals_with_pipeline_source = liq_terminals_with_pipeline_source
        self.liq_terminals_with_pipeline_sink = liq_terminals_with_pipeline_sink

        self.env.process(self.process(seed, self.run_id))

    def process(self, seed, run_id):
        """
        Process to handle the truck loading and unloading at the terminal.
        Args:
            seed (int): Random seed for reproducibility.
            run_id (int): Unique identifier for the simulation run.
        Yields:
            simpy.Process: The truck loading and unloading process.
        Returns:
            None
        """
        import_terminal = get_value_by_terminal(
            self.terminal_data, self.terminal_type, self.terminal_id, 'import')
        export_terminal = get_value_by_terminal(
            self.terminal_data, self.terminal_type, self.terminal_id, 'export')
        action = actions.get((import_terminal, export_terminal))

        start_time = self.env.now

        if self.terminal_type == "Container":
            yield self.env.process(self.container_truck(run_id=run_id, truck_id=f"C.{self.terminal_id}:{self.truck_id}", load_amount=self.container_amount[0], unload_amount=self.container_amount[1], terminal_id=1, action=action, seed=seed))
        elif self.terminal_type == "Liquid":
            # if action == "both":
            #     action = random.choice(["load", "unload"]) # new
            yield self.env.process(self.tanker_truck(truck_id=f"L.{self.terminal_id}:{self.truck_id}", amount=self.liquid_amount, terminal_id=1, action=action, seed=seed))
        elif self.terminal_type == "DryBulk":
            # if action == "both":
            #     action = random.choice(["load", "unload"]) # new
            yield self.env.process(self.drybulk_truck(truck_id=f"D.{self.terminal_id}:{self.truck_id}", amount=self.drybulk_amount, terminal_id=1, action=action, seed=seed))

        dwell_time = self.env.now - start_time

        # log the truck id, start time, dwell time, terminal id, terminal type
        log_line(self.run_id, "truck_data.txt", f'truck_id: {self.truck_id}, start_time: {start_time}, dwell_time: {dwell_time}, terminal_id: {self.terminal_id}, terminal_type: {self.terminal_type}')


    def tanker_truck(self, truck_id, amount, terminal_id, action, seed):
        """
        Process to handle the tanker truck loading and unloading at the liquid terminal.
        Args:
            truck_id (str): Unique identifier for the truck.
            amount (int): Amount of liquid to load or unload.
            terminal_id (int): Unique identifier for the terminal.
            action (str): Action to perform, either "load" or "unload".
            seed (int): Random seed for reproducibility.
        Yields:
            simpy.Process: The tanker truck loading and unloading process.
        Returns:
            None
        """
        random.seed(seed)
        self.chassis_bays_utilization["Liquid"][self.terminal_id].append(
            (self.env.now, self.loading_bays.count / self.loading_bays.capacity))
        with self.loading_bays.request() as req:
            yield req
            TRUCK_WAITING_TIME = random.uniform(
                TRUCK_WAITING_TIME_MIN, TRUCK_WAITING_TIME_MAX)
            yield self.env.timeout(TRUCK_WAITING_TIME)
            TRUCK_PUMP_RATE = self.transfer_rate

            if action == "load":
                yield self.port_tanks.get(amount)
                load_time = amount / TRUCK_PUMP_RATE
                yield self.env.timeout(load_time)
            elif action == "unload":
                yield self.port_tanks.put(amount)
                unload_time = amount / TRUCK_PUMP_RATE
                yield self.env.timeout(unload_time)

        if self.port_tanks.level + amount >= 0.9 * self.port_tanks.capacity:
            if terminal_id in self.liq_terminals_with_pipeline_sink:
                Pipeline(self.run_id, self.env, self.port_tanks,
                         mode='sink', rate=constants.PIPELINE_RATE)
                with open(f'.{self.run_id}/logs/force_action.txt', 'a') as f:
                    f.write(
                        f"Pipeline from source sink activated {self.env.now}")
                    f.write('\n')

        if self.port_tanks.level - amount <= 0.1 * self.port_tanks.capacity:
            if terminal_id in self.liq_terminals_with_pipeline_source:
                Pipeline(self.run_id, self.env, self.port_tanks,
                         mode='source', rate=constants.PIPELINE_RATE)
                with open(f'.{self.run_id}/logs/force_action.txt', 'a') as f:
                    f.write(
                        f"Pipeline from source source activated {self.env.now}")
                    f.write('\n')

    def drybulk_truck(self, truck_id, amount, terminal_id, action, seed):
        """
        Process to handle the dry bulk truck loading and unloading at the dry bulk terminal.    
        Args:
            truck_id (str): Unique identifier for the truck.
            amount (int): Amount of dry bulk to load or unload.
            terminal_id (int): Unique identifier for the terminal.
            action (str): Action to perform, either "load" or "unload".
            seed (int): Random seed for reproducibility.
        Yields:
            simpy.Process: The dry bulk truck loading and unloading process.
        Returns:
            None
        """
        random.seed(seed)
        self.chassis_bays_utilization["DryBulk"][self.terminal_id].append(
            (self.env.now, self.drybulk_bays.count / self.drybulk_bays.capacity))
        start_time = self.env.now
        with self.drybulk_bays.request() as req:
            yield req
            TRUCK_WAITING_TIME = random.uniform(
                TRUCK_WAITING_TIME_MIN, TRUCK_WAITING_TIME_MAX)
            yield self.env.timeout(TRUCK_WAITING_TIME)
            TRUCK_TRANSFER_RATE = self.transfer_rate

            if action == "load":
                yield self.port_silos.get(amount)
                load_time = amount / TRUCK_TRANSFER_RATE
                yield self.env.timeout(load_time)
            elif action == "unload":
                yield self.port_silos.put(amount)
                unload_time = amount / TRUCK_TRANSFER_RATE
                yield self.env.timeout(unload_time)

    def container_truck(self, run_id, truck_id, load_amount, unload_amount, terminal_id, action, seed):
        """
        Process to handle the container truck loading and unloading at the container terminal.
        Args:
            run_id (int): Unique identifier for the simulation run.
            truck_id (str): Unique identifier for the truck.
            load_amount (int): Amount of containers to load.
            unload_amount (int): Amount of containers to unload.
            terminal_id (int): Unique identifier for the terminal.
            action (str): Action to perform, either "load", "unload", or "both".
            seed (int): Random seed for reproducibility.
        Yields:
            simpy.Process: The container truck loading and unloading process.
        Returns:
            None
        """
        random.seed(seed)
        start_time = self.env.now
        self.chassis_bays_utilization["Container"][self.terminal_id].append(
            (self.env.now, len(self.truck_chassis.items) / self.truck_chassis.capacity))
        TRUCK_WAITING_TIME = random.uniform(
            TRUCK_WAITING_TIME_MIN, TRUCK_WAITING_TIME_MAX)
        yield self.env.timeout(TRUCK_WAITING_TIME)

        force = False
        if constants.CTR_TRUCK_OVERRIDE:
            if len(self.port_yard.items) <= 0.05 * self.port_yard.capacity:
                action = "unload"
                force = True
                with open(f'.{run_id}/logs/force_action.txt', 'a') as f:
                    f.write(
                        f"Container truck force unloading at {self.env.now}")
                    f.write('\n')
            elif len(self.port_yard.items) >= 0.95 * self.port_yard.capacity:
                action = "load"
                force = True
                with open(f'.{run_id}/logs/force_action.txt', 'a') as f:
                    f.write(f"Container truck force loading at {self.env.now}")
                    f.write('\n')
        if action == "both":
            force = False
            action = random.choice(["load", "unload"])

        if action == "load":
            if not force:
                yield self.truck_chassis.get()
            for _ in range(load_amount):
                yield self.port_yard.get()
                load_time = 1 / self.transfer_rate
                yield self.env.timeout(load_time)
        if action == "unload":
            for _ in range(unload_amount):
                yield self.port_yard.put(1)
                unload_time = 1 / self.transfer_rate
                yield self.env.timeout(unload_time)
            if not force:
                yield self.truck_chassis.put(1)
