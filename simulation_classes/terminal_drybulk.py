"""
Uses dry bulk terminal class objects to create dry bulk terminal processes.
Vessel arrivals, berth and conveyor allocation, cargo unloading, cargo loading, vessel waiting, and vessel departure are simulated.
"""
from simulation_analysis.results import update_ship_logs
from simulation_handler.helpers import get_value_by_terminal
from simulation_handler.helpers import is_daytime
import constants

TIME_COMMON_CHANNEL = constants.TIME_COMMON_CHANNEL
TIME_FOR_TUG_STEER = constants.TIME_FOR_TUG_STEER
TIME_FOR_UTURN = constants.TIME_FOR_UTURN
DRY_BULK_EFFICIENCY = constants.DRYBULK_TERMINAL_EFFICIENCY

nan = float('nan')


class DryBulkTerminal:
    """
    Represents a dry bulk terminal and the process of handling arriving ships.
    This class simulates the arrival, berthing, unloading, loading, waiting, and departure of dry bulk ships at the terminal.
    It manages the allocation of berths and conveyors, handles cargo operations, and logs events during the simulation.
    The process includes the following steps:
    1. Ship arrival at the terminal.
    2. Anchorage waiting period.
    3. Berth allocation for the ship.
    4. Channel process for entering the terminal.
    5. Docking the ship to the berth.
    6. Unloading dry bulk cargo from the ship.
    7. Loading dry bulk cargo onto the ship.
    8. Waiting at the terminal until the ship is ready to depart.
    9. Detaching the ship from the berth and departing from the terminal.
    10. Channel process for leaving the terminal.
    The class also logs events and maintains ship logs throughout the process.

    Args:
        env (SimPy.Environment): The simulation environment.
        chassis_bays_utilization (float): Utilization of chassis bays.
        run_id (int): Unique identifier for the simulation run.
        channel (Channel): The channel object for managing ship movements.  
        day_pilots (simpy.Container): Simpy container of available day pilots.
        night_pilots (simpy.Container) : Simpy container of available night pilots.
        tugboats (simpy.Container): Simpy container of available tugboats.
        ship_info (dict): Information about the ships.
        last_section (str): The last section of the channel.
        selected_terminal (str): The terminal selected for the ship.
        id (int): Unique identifier for the ship.
        ship_type (str): Type of the ship (e.g., "DryBulk").
        draft (float): Draft of the ship.
        width (float): Width of the ship.
        unload_time (float): Time taken to unload cargo (in hours).
        load_time (float): Time taken to load cargo (in hours).
        unload_tons (float): Amount of cargo to unload (in tons).
        load_tons (float): Amount of cargo to load (in tons).
        events (list): List to store events during the simulation.
        ship_logs (list): List to store logs of the ship's activities.
        port_berths (simpy.Resource): Resource for managing port berths.
        port_silos (simpy.Resource): Resource for managing port silos.
        SHIPS_IN_CHANNEL (int): Number of ships allowed in the channel.
        SHIPS_IN_ANCHORAGE (int): Number of ships allowed in the anchorage.
        terminal_data (dict): Data related to the terminal.
        """
    def __init__(self, env, chassis_bays_utilization, run_id, channel, day_pilots, night_pilots, tugboats, ship_info, last_section, selected_terminal, id, ship_type, draft, width, unload_time, load_time, unload_tons, load_tons, events, ship_logs, port_berths, port_silos, SHIPS_IN_CHANNEL, SHIPS_IN_ANCHORAGE, terminal_data):
        # Initialize the environment
        self.env = env
        self.run_id = run_id
        self.SHIPS_IN_CHANNEL = SHIPS_IN_CHANNEL
        self.SHIPS_IN_ANCHORAGE = SHIPS_IN_ANCHORAGE
        self.terminal_data = terminal_data

        # Initalise the channel
        self.channel = channel
        self.ship_info = ship_info
        self.last_section = last_section

        # day_pilots, night_pilots and tugboats
        self.day_pilots, self.night_pilots = day_pilots, night_pilots
        self.tugboats = tugboats

        # Set vessel and terminal attributes
        self.id = id
        self.selected_terminal = selected_terminal
        self.ship_type = ship_type
        self.num_conveyors = get_value_by_terminal(
            self.terminal_data, 'DryBulk', selected_terminal, 'transfer units per berth')
        self.unload_tons = unload_tons
        self.load_tons = load_tons
        self.time_for_uturn_min, self.time_for_uturn_max = TIME_FOR_UTURN

        self.draft = draft
        self.width = width
        self.unload_time = unload_time
        self.load_time = load_time

        # wait time proxy
        self.cargo_wait_time = 0

        # The allocation of conveyors and berth will be done during the process
        self.current_berth = None
        self.allocated_conveyors = []

        # Get the ship logs and events
        self.events = events
        self.ship_logs = ship_logs

        # Allocate berth and conveyors
        self.port_berths = port_berths
        self.port_silos = port_silos

        # Start the process
        self.env.process(self.process())

    def process(self):
        """
        The main process for the dry bulk terminal, simulating the arrival, berthing, unloading, loading, waiting, and departure of ships.
        This method handles the entire lifecycle of a dry bulk ship at the terminal.
        """

        self.ship_logs = update_ship_logs(self.ship_logs, "D", self.id, self.selected_terminal, ship_start_time=nan, time_to_get_berth=nan, time_for_restriction_in=nan, time_to_get_pilot_in=nan,
                                          time_to_get_tugs_in=nan, time_to_common_channel_in=nan, time_to_travel_channel_in=nan, time_to_tug_steer_in=nan, unloading_time=nan, loading_time=nan, waiting_time=nan,
                                          departure_time=nan, time_to_get_pilot_out=nan, time_to_get_tugs_out=nan, time_to_tug_steer_out=nan, time_for_uturn=nan,
                                          time_to_travel_channel_out=nan, time_to_common_channel_out=nan, ship_end_time=nan)

        # ship arrival
        ship_start_time = self.env.now
        update_ship_logs(self.ship_logs, "D", self.id,
                         self.selected_terminal, ship_start_time=ship_start_time)
        self.events.append((f"Ship_{self.id}", "DryBulk", f"D.{self.selected_terminal}",
                           "arrive", self.env.now, f"Ship {self.id} arrived at the port"))

        # Anchorage waiting
        yield self.env.timeout(constants.ANCHORAGE_WAITING_DRYBULK)

        # Berth allocation
        start_time = self.env.now
        yield self.env.process(self.arrive_and_berth())
        time_to_get_berth = self.env.now - ship_start_time
        update_ship_logs(self.ship_logs, "D", self.id,
                         self.selected_terminal, time_to_get_berth=time_to_get_berth)

        # Channel process (in)
        self.day = is_daytime(self.env.now, 7, 19)
        yield self.env.process(self.channel.channel_process(self.ship_info, self.day, self.last_section, self.id, self.selected_terminal, "D", self.run_id))

        # Terminal process
        self.events.append((f"Ship_{self.id}", "DryBulk", f"L.{self.selected_terminal}", "dock", self.env.now,
                           f"Ship {self.id} docked to berth {self.current_berth.id} with waiting time {'N/A'}"))

        start_time = self.env.now
        yield self.env.process(self.unload())
        unloading_time = self.env.now - start_time
        update_ship_logs(self.ship_logs, "D", self.id,
                         self.selected_terminal, unloading_time=unloading_time)

        start_time = self.env.now
        yield self.env.process(self.load())
        loading_time = self.env.now - start_time
        update_ship_logs(self.ship_logs, "D", self.id,
                         self.selected_terminal, loading_time=loading_time)

        start_time = self.env.now
        yield self.env.process(self.wait())
        waiting_time = self.env.now - start_time
        update_ship_logs(self.ship_logs, "D", self.id,
                         self.selected_terminal, waiting_time=waiting_time)

        start_time = self.env.now
        yield self.env.process(self.detach_and_depart())
        departure_time = self.env.now - start_time
        update_ship_logs(self.ship_logs, "D", self.id,
                         self.selected_terminal, departure_time=departure_time)

        self.events.append((f"Ship_{self.id}", "DryBulk", f"D.{self.selected_terminal}", "undock", self.env.now,
                           f"Ship {self.id} undocked to berth {self.current_berth.id} with waiting time {'N/A'}"))

        # Channel process (out)
        self.day = is_daytime(self.env.now, 7, 19)
        yield self.env.process(self.channel.channel_process(self.ship_info, self.day, self.last_section, self.id, self.selected_terminal, "D", self.run_id))

        ship_end_time = self.env.now
        update_ship_logs(self.ship_logs, "D", self.id,
                         self.selected_terminal, ship_end_time=ship_end_time)
        self.events.append((f"Ship_{self.id}", "DryBulk", f"D.{self.selected_terminal}", "depart", self.env.now,
                           f"Ship {self.id} departed from berth {self.current_berth.id} with waiting time {'N/A'}"))

    def arrive_and_berth(self):
        """
        Represents the process of a vessel arriving at the terminal and seizing a berth.
        This method handles the allocation of a berth for the vessel and prepares it for unloading and loading operations.
        """
        self.current_berth = yield self.port_berths.get()

    def unload(self):
        """
        Represents the unloading of dry bulk cargo from the vessel at the terminal.
        This method allocates the required number of conveyors, unloads the cargo, and stores it in the port silos.
        It also handles the waiting time until the unloading is completed, ensuring that it only proceeds during daytime hours.
        The unloading process is divided into several steps:
        1. Check if the terminal has the required import conveyors.
        2. Allocate the conveyors for unloading.
        3. Unload the dry bulk cargo from the vessel.   
        4. Wait until the unloading is completed, ensuring it only occurs during daytime hours.
        5. Store the unloaded dry bulk cargo in the port silos.
        """
        # Request for the required number of conveyors
        import_terminal = get_value_by_terminal(
            self.terminal_data, 'DryBulk', self.selected_terminal, 'import')
        if import_terminal:
            for _ in range(self.num_conveyors):
                conveyor = yield self.current_berth.conveyors.get()
                self.allocated_conveyors.append(conveyor)
            self.events.append((f"Ship_{self.id}", "DryBulk", f"D.{self.selected_terminal}",
                               "all_conveyors_attached", self.env.now, f"Ship {self.id} conveyors allocated"))
            # Unload dry bulk cargo
            total_unload_time = self.unload_tons * self.unload_time / self.num_conveyors
            self.cargo_wait_time += total_unload_time
            # keep waiting until day time
            while total_unload_time > 0:
                if not is_daytime(self.env.now, start=7, end=19):
                    yield self.env.timeout(1)
                else:
                    yield self.env.timeout(1)
                    total_unload_time -= 1
            # yield self.env.timeout(total_unload_time)
            self.events.append(("Ship_{}".format(self.id), "DryBulk", f"D.{self.selected_terminal}",
                               "all_bulk_unloaded", self.env.now, "Ship {} unloaded all dry bulk".format(self.id)))
            # Put dry bulk batches in silos
            yield self.port_silos.put(self.unload_tons)
            self.events.append(("Ship_{}".format(self.id), "DryBulk", f"D.{self.selected_terminal}",
                               "all_bulk_stored", self.env.now, "Ship {} unloaded all dry bulk".format(self.id)))
        else:
            self.events.append((f"Ship_{self.id}", "DryBulk", f"D.{self.selected_terminal}",
                               "all_conveyors_attached", self.env.now, f"Ship {self.id} conveyors allocated"))
            self.events.append(("Ship_{}".format(self.id), "DryBulk", f"D.{self.selected_terminal}",
                               "all_bulk_unloaded", self.env.now, "Ship {} unloaded all dry bulk".format(self.id)))
            self.events.append(("Ship_{}".format(self.id), "DryBulk", f"D.{self.selected_terminal}",
                               "all_bulk_stored", self.env.now, "Ship {} unloaded all dry bulk".format(self.id)))

    def load(self):
        """
        Represents the loading of dry bulk cargo onto the vessel at the terminal.
        This method allocates the required number of conveyors, loads the cargo onto the vessel, and updates the port silos.
        It also handles the waiting time until the loading is completed, ensuring that it only proceeds during daytime hours.
        The loading process is divided into several steps:
        1. Check if the terminal has the required export conveyors.
        2. Allocate the conveyors for loading.
        3. Load the dry bulk cargo onto the vessel.
        4. Wait until the loading is completed, ensuring it only occurs during daytime hours.
        5. Update the port silos to reflect the loaded cargo.
        """
        export_terminal = get_value_by_terminal(
            self.terminal_data, 'DryBulk', self.selected_terminal, 'export')
        if export_terminal:
            # Represents dry bulk cargo being loaded onto the vessel at the input conveyor rate
            total_load_time = self.load_tons * self.load_time / self.num_conveyors
            # Load dry bulk cargo
            yield self.port_silos.get(self.load_tons)
            self.cargo_wait_time += total_load_time
            # keep waiting until day time
            while total_load_time > 0:
                if not is_daytime(self.env.now, start=10, end=17):
                    yield self.env.timeout(1)
                else:
                    yield self.env.timeout(1)
                    total_load_time -= 1
            self.events.append(("Ship_{}".format(self.id), "DryBulk", f"D.{self.selected_terminal}",
                               "all_bulk_loaded", self.env.now, "Ship {} loaded all dry bulk".format(self.id)))
        else:
            self.events.append(("Ship_{}".format(self.id), "DryBulk", f"D.{self.selected_terminal}",
                               "all_bulk_loaded", self.env.now, "Ship {} loaded all dry bulk".format(self.id)))

    def wait(self):
        """
        Represents the waiting time for the vessel at the port after unloading and loading operations.
        This method calculates the waiting time based on the dry bulk efficiency and the cargo wait time.
        It ensures that the vessel waits at the port until it is ready to depart.
        """
        port_waiting_time = (1/DRY_BULK_EFFICIENCY - 1) * self.cargo_wait_time
        yield self.env.timeout(port_waiting_time)
        self.events.append(("Ship_{}".format(self.id), "DryBulk", f"D.{self.selected_terminal}",
                           "wait", self.env.now, "Ship {} waited at the port".format(self.id)))

    def detach_and_depart(self):
        """
        Represents the process of detaching the vessel from the berth and departing from the port.
        This method handles the release of the allocated conveyors, updates the ship logs, and releases the berth.
        It also logs the events related to the vessel's departure.
        """

        # Release the conveyor belts
        for conveyor in self.allocated_conveyors:
            yield self.current_berth.conveyors.put(conveyor)
            self.events.append(("Ship_{}".format(self.id), "DryBulk", f"D.{self.selected_terminal}", "all_conveyors_disconnected",
                               self.env.now, "Ship {} disconnected from conveyor {}".format(self.id, conveyor.id)))

        # Depart from the berth
        yield self.port_berths.put(self.current_berth)
