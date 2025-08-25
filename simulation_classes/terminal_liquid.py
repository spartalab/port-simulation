"""
Uses liquid bulk terminal class objects to create liquid terminal processes.
Vessel arrivals, berth and pipeline allocation, cargo unloading, cargo loading, vessel waiting, and vessel departure are simulated. 
"""
from simulation_analysis.results import update_ship_logs
from simulation_handler.helpers import get_value_by_terminal, is_daytime
from simulation_classes.pipeline import Pipeline
import constants
import simpy

LIQUID_TERMINAL_EFFICIENCY = constants.LIQ_TERMINAL_EFFICIENCY
TIME_COMMON_CHANNEL = constants.TIME_COMMON_CHANNEL
TIME_FOR_TUG_STEER = constants.TIME_FOR_TUG_STEER
TIME_FOR_UTURN = constants.TIME_FOR_UTURN

nan = float('nan')


class LiquidTerminal:
    """
    LiquidTerminal class simulates the liquid bulk terminal operations, including ship arrivals, berthing, unloading, loading, waiting, and departure.
    It manages the allocation of berths and pipelines, handles cargo operations, and logs events related to the terminal operations.
    This class is initialized with various parameters such as the simulation environment, channel, pilots, tugboats, ship information, terminal data, and more.
    The processes include the following steps:
    1. Ship arrival and anchorage waiting.
    2. Berth allocation.
    3. Channel process for entering the terminal.
    4. Unloading of liquid bulk cargo.
    5. Loading of liquid bulk cargo.
    6. Waiting time at the terminal.
    7. Detachment from the berth and departure from the terminal.
    8. Channel process for exiting the terminal.
    The class also handles the allocation of pipelines based on the terminal type (import/export) and manages the storage of cargo in port tanks.
    It logs all events and updates ship logs throughout the process.
    
    Args:
        env (simpy.Environment): The simulation environment.
        chassis_bays_utilization (float): Utilization of chassis bays.
        run_id (str): Unique identifier for the simulation run.
        channel (Channel): The channel object for managing ship movements.
        day_pilots (simpy.Container): Simpy container for day pilots available.
        night_pilots (simpy.Container) : Simpy container for night pilots available.    
        tugboats (simpy.Container): Simpy container for tugboats available.
        ship_info (dict): Information about the ships.
        last_section (str): The last section of the channel.
        selected_terminal (str): The terminal selected for the ship.
        id (int): Unique identifier for the ship.
        ship_type (str): Type of the ship (e.g., "Liquid").
        draft (float): Draft of the ship.
        width (float): Width of the ship.
        unload_time (float): Time taken to unload the ship.
        load_time (float): Time taken to load the ship.
        unload_tons (float): Amount of cargo to unload from the ship.
        load_tons (float): Amount of cargo to load onto the ship.
        events (list): List to store events related to the ship.
        ship_logs (dict): Dictionary to store logs for the ship.
        port_berths (simpy.Resource): Resource for managing port berths.
        port_tanks (simpy.Container): Container for managing port tanks.
        SHIPS_IN_CHANNEL (int): Number of ships currently in the channel.
        SHIPS_IN_ANCHORAGE (int): Number of ships currently in the anchorage.
        terminal_data (dict): Data related to the terminal.
        liq_terminals_with_pipeline_source (list): List of liquid terminals with pipeline source.
        liq_terminals_with_pipeline_sink (list): List of liquid terminals with pipeline sink.
    """
    def __init__(self, env, chassis_bays_utilization, run_id, channel, day_pilots, night_pilots, tugboats, ship_info, last_section, selected_terminal, id, ship_type, draft, width, unload_time, load_time, unload_tons, load_tons, events, ship_logs, port_berths, port_tanks, SHIPS_IN_CHANNEL, SHIPS_IN_ANCHORAGE, terminal_data, liq_terminals_with_pipeline_source, liq_terminals_with_pipeline_sink):
        # Initialize the environment
        self.env = env
        self.run_id = run_id
        self.SHIPS_IN_CHANNEL = SHIPS_IN_CHANNEL
        self.SHIPS_IN_ANCHORAGE = SHIPS_IN_ANCHORAGE
        self.terminal_data = terminal_data

        # Initialize the channel
        self.channel = channel
        self.ship_info = ship_info
        self.last_section = last_section

        # Pilots and tugboats
        self.day_pilots, self.night_pilots = day_pilots, night_pilots
        self.tugboats = tugboats

        # Set vessel and terminal attributes
        self.id = id
        self.selected_terminal = selected_terminal
        self.ship_type = ship_type
        self.num_pipelines = get_value_by_terminal(
            self.terminal_data, 'Liquid', selected_terminal, 'transfer units per berth')

        # Set vessel dimensions and cargo handling parameters
        self.unload_tons = unload_tons
        self.load_tons = load_tons
        self.draft = draft
        self.width = width
        self.unload_time = unload_time
        self.load_time = load_time
        self.day = None

        # Cargo wait time
        self.cargo_wait_time = 0

        # Pipelines
        self.liq_terminals_with_pipeline_source = liq_terminals_with_pipeline_source
        self.liq_terminals_with_pipeline_sink = liq_terminals_with_pipeline_sink
        self.time_for_uturn_min, self.time_for_uturn_max = TIME_FOR_UTURN

        # The allocation of pipelines and berth will be done during the process
        self.current_berth = None
        self.allocated_pipelines = []

        # Get the ship logs and events
        self.events = events
        self.ship_logs = ship_logs

        # Allocate berth and pipelines
        self.port_berths = port_berths
        self.port_tanks = port_tanks

        # Start the process
        self.env.process(self.process())

    def process(self):
        """
        Process for the liquid bulk terminal, simulating the arrival, berthing, unloading, loading, waiting, and departure of ships.
        This method handles the entire lifecycle of a ship at the terminal, including logging events and updating ship logs."""

        self.ship_logs = update_ship_logs(self.ship_logs, "L", self.id, self.selected_terminal, ship_start_time=nan, time_to_get_berth=nan, time_for_restriction_in=nan, time_to_get_pilot_in=nan,
                                          time_to_get_tugs_in=nan, time_to_common_channel_in=nan, time_to_travel_channel_in=nan, time_to_tug_steer_in=nan, unloading_time=nan, loading_time=nan, waiting_time=nan,
                                          departure_time=nan, time_to_get_pilot_out=nan, time_to_get_tugs_out=nan, time_to_tug_steer_out=nan, time_for_uturn=nan,
                                          time_to_travel_channel_out=nan, time_to_common_channel_out=nan, ship_end_time=nan)

        # ship arrival
        ship_start_time = self.env.now
        update_ship_logs(self.ship_logs, "L", self.id,
                         self.selected_terminal, ship_start_time=ship_start_time)
        self.events.append((f"Ship_{self.id}", "Liquid", f"L.{self.selected_terminal}",
                           "arrive", self.env.now, f"Ship {self.id} arrived at the port"))

        # Anchorage waiting
        yield self.env.timeout(constants.ANCHORAGE_WAITING_LIQUID)

        # Berth allocation
        start_time = self.env.now
        yield self.env.process(self.arrive_and_berth())
        time_to_get_berth = self.env.now - ship_start_time
        update_ship_logs(self.ship_logs, "L", self.id,
                         self.selected_terminal, time_to_get_berth=time_to_get_berth)

        # Channel process (in)
        self.day = is_daytime(self.env.now, 7, 19)
        yield self.env.process(self.channel.channel_process(self.ship_info, self.day, self.last_section, self.id, self.selected_terminal, "L", self.run_id))

        # Terminal process
        self.events.append((f"Ship_{self.id}", "Liquid", f"L.{self.selected_terminal}", "dock", self.env.now,
                           f"Ship {self.id} docked to berth {self.current_berth.id} with waiting time {'N/A'}"))
        start_time = self.env.now
        yield self.env.process(self.unload())
        unloading_time = self.env.now - start_time
        update_ship_logs(self.ship_logs, "L", self.id,
                         self.selected_terminal, unloading_time=unloading_time)

        start_time = self.env.now
        yield self.env.process(self.load())
        loading_time = self.env.now - start_time
        update_ship_logs(self.ship_logs, "L", self.id,
                         self.selected_terminal, loading_time=loading_time)

        start_time = self.env.now
        yield self.env.process(self.wait())
        waiting_time = self.env.now - start_time
        update_ship_logs(self.ship_logs, "L", self.id,
                         self.selected_terminal, waiting_time=waiting_time)

        start_time = self.env.now
        yield self.env.process(self.detach_and_depart())
        departure_time = self.env.now - start_time
        update_ship_logs(self.ship_logs, "L", self.id,
                         self.selected_terminal, departure_time=departure_time)

        self.events.append((f"Ship_{self.id}", "Liquid", f"L.{self.selected_terminal}", "undock", self.env.now,
                           f"Ship {self.id} undocked to berth {self.current_berth.id} with waiting time {'N/A'}"))

        # Channel process (out)
        self.day = is_daytime(self.env.now, 7, 19)
        yield self.env.process(self.channel.channel_process(self.ship_info, self.day, self.last_section, self.id, self.selected_terminal, "L", self.run_id))
        ship_end_time = self.env.now
        update_ship_logs(self.ship_logs, "L", self.id,
                         self.selected_terminal, ship_end_time=ship_end_time)
        self.events.append((f"Ship_{self.id}", "Liquid", f"L.{self.selected_terminal}", "depart", self.env.now,
                           f"Ship {self.id} departed from berth {self.current_berth.id} with waiting time {'N/A'}"))

    def arrive_and_berth(self):
        """
        Represents the arrival of the ship at the terminal and allocation of a berth.
        This method waits for a berth to become available and allocates it to the ship.
        """
        # Wait for a berth to become available
        self.current_berth = yield self.port_berths.get()

    def unload(self):
        """
        Represents the unloading of liquid bulk cargo from the ship at the terminal.
        This method checks if the terminal is exporting or importing, allocates pipelines, and unloads the cargo.
        It also handles the storage of the unloaded cargo in the port tanks and ensures that the tanks have enough capacity.
        If the port tanks are full, it activates a pipeline sink to empty the tanks.
        If the port tanks are empty, it activates a pipeline source to fill the tanks.
        It also ensures that the unloading process does not exceed the tank capacity.
        The method logs events related to the unloading process.
        """

        # check if terminal is exporting or importing
        import_terminal = get_value_by_terminal(
            self.terminal_data, 'Liquid', self.selected_terminal, 'import')
        if import_terminal:
            # Request for the required number of pipelines
            for _ in range(self.num_pipelines):
                pipeline = yield self.current_berth.pipelines.get()
                self.allocated_pipelines.append(pipeline)
            self.events.append((f"Ship_{self.id}", "Liquid", f"L.{self.selected_terminal}",
                               "all_pipelines_attached", self.env.now, f"Ship {self.id} pipelines allocated"))

            # Unload liquid bulk cargo
            total_unload_time = self.unload_tons * self.unload_time / self.num_pipelines
            self.cargo_wait_time += total_unload_time
            yield self.env.timeout(total_unload_time)
            self.events.append(("Tanker_{}".format(self.id), "Liquid", f"L.{self.selected_terminal}",
                               "all_liquid_unloaded", self.env.now, "Tanker {} unloaded all liquid".format(self.id)))

            # Put liquid batches in tanks
            if self.port_tanks.level + self.unload_tons >= 0.9 * self.port_tanks.capacity:
                if self.selected_terminal in self.liq_terminals_with_pipeline_sink:
                    Pipeline(self.run_id, self.env, self.port_tanks,
                             mode='sink', rate=constants.PIPELINE_RATE)
                    with open(f'.{self.run_id}/logs/force_action.txt', 'a') as f:
                        f.write(f"Pipeline sink activated {self.env.now}")
                        f.write('\n')

            # Do not offload until there is enough space in the port tanks (New feature added on 2025-03-24) #TODO: Check if this is the right way to do it
            while self.port_tanks.level + self.unload_tons >= self.port_tanks.capacity:
                yield self.env.timeout(1)

            yield self.port_tanks.put(self.unload_tons)

            self.events.append(("Tanker_{}".format(self.id), "Liquid", f"L.{self.selected_terminal}",
                               "all_liquid_stored", self.env.now, "Tanker {} unloaded all liquid".format(self.id)))
        else:
            self.events.append((f"Ship_{self.id}", "Liquid", f"L.{self.selected_terminal}",
                               "all_pipelines_attached", self.env.now, f"Ship {self.id} pipelines allocated"))
            self.events.append(("Tanker_{}".format(self.id), "Liquid", f"L.{self.selected_terminal}",
                               "all_liquid_unloaded", self.env.now, "Tanker {} unloaded all liquid".format(self.id)))
            self.events.append(("Tanker_{}".format(self.id), "Liquid", f"L.{self.selected_terminal}",
                               "all_liquid_stored", self.env.now, "Tanker {} unloaded all liquid".format(self.id)))

    def load(self):
        """
        Represents the loading of liquid bulk cargo onto the ship at the terminal.
        This method checks if the terminal is exporting or importing, allocates pipelines, and loads the cargo.
        It also handles the storage of the loaded cargo in the port tanks and ensures that the tanks have enough capacity.
        If the port tanks are empty, it activates a pipeline source to fill the tanks.
        It also ensures that the loading process does not exceed the tank capacity.
        The method logs events related to the loading process.
        """

        export_terminal = get_value_by_terminal(
            self.terminal_data, 'Liquid', self.selected_terminal, 'export')
        if export_terminal:
            # Represents liquid bulk cargo being loaded onto the vessel at the input pump rate
            total_load_time = self.load_tons * self.load_time / self.num_pipelines
            # Load liquid bulk cargo
            if self.port_tanks.level - self.load_tons <= 0.1 * self.port_tanks.capacity:
                if self.selected_terminal in self.liq_terminals_with_pipeline_source:
                    Pipeline(self.run_id, self.env, self.port_tanks,
                             mode='source', rate=constants.PIPELINE_RATE)
                    with open(f'.{self.run_id}/logs/force_action.txt', 'a') as f:
                        f.write(f"Pipeline source activated {self.env.now}")
                        f.write('\n')

            # Do not load until there is enough liquid in the port tanks (New feature added on 2025-03-24) #TODO: Check if this is the right way to do it
            while self.port_tanks.level - self.load_tons <= 0.0 * self.port_tanks.capacity:
                yield self.env.timeout(1)

            yield self.port_tanks.get(self.load_tons)
            self.cargo_wait_time += total_load_time
            yield self.env.timeout(total_load_time)
            self.events.append(("Tanker_{}".format(self.id), "Liquid", f"L.{self.selected_terminal}",
                               "all_liquid_loaded", self.env.now, "Tanker {} loaded all liquid".format(self.id)))
        else:
            self.events.append(("Tanker_{}".format(self.id), "Liquid", f"L.{self.selected_terminal}",
                               "all_liquid_loaded", self.env.now, "Tanker {} loaded all liquid".format(self.id)))

    def wait(self):
        """
        Represents the waiting time of the vessel at the terminal after loading or unloading.
        This method simulates the waiting time before the vessel departs the terminal.
        It calculates the waiting time based on the liquid terminal efficiency and the cargo wait time.
        The method logs events related to the waiting process.
        """
        port_waiting_time = (
            1/LIQUID_TERMINAL_EFFICIENCY - 1) * self.cargo_wait_time
        yield self.env.timeout(port_waiting_time)
        self.events.append(("Tanker_{}".format(self.id), "Liquid", f"L.{self.selected_terminal}",
                           "wait", self.env.now, "Tanker {} waited at the port".format(self.id)))

    def detach_and_depart(self):
        """
        Represents the detachment of the ship from the berth and its departure from the terminal.
        This method releases the allocated pipelines, puts the berth back into the port berths resource, and logs events related to the detachment and departure process.
        """
        # Release the pipelines
        for pipeline in self.allocated_pipelines:
            yield self.current_berth.pipelines.put(pipeline)
            self.events.append(("Tanker_{}".format(self.id), "Liquid", f"L.{self.selected_terminal}", "all_pipelines_disconnected",
                               self.env.now, "Tanker {} disconnected from pipeline {}".format(self.id, pipeline.id)))

        # Depart from the berth
        yield self.port_berths.put(self.current_berth)
