"""
Uses container terminal class objects to create container terminal processes.
Vessel arrivals, berth and crane allocation, container unloading, container loading, vessel waiting, and vessel departure are simulated.
"""
import constants
from simulation_analysis.results import update_ship_logs
from simulation_handler.helpers import get_value_by_terminal, is_daytime

TIME_COMMON_CHANNEL = constants.TIME_COMMON_CHANNEL
TIME_FOR_TUG_STEER = constants.TIME_FOR_TUG_STEER
TIME_FOR_UTURN = constants.TIME_FOR_UTURN
CONTAINER_TERMINAL_EFFICIENCY = constants.CTR_TERMINAL_EFFICIENCY

nan = float('nan')


class ContainerTerminal:
    """
    Class to simulate the container terminal and the processes of arriving ships.
    This class handles the docking, unloading, loading, waiting, and departure of container ships.
    It also manages the allocation of berths and cranes, and the interaction with the channel for ship movements.   
    The processes include the following steps:
    1. Ship arrival at the port.
    2. Anchorage waiting.
    3. Berth allocation.
    4. Channel process (inbound).
    5. Docking the ship to a berth.
    6. Unloading containers from the ship.
    7. Loading containers onto the ship.
    8. Waiting at the port after unloading and loading operations.
    9. Detaching from the berth and departing from the port.
    10. Channel process (outbound).
    11. Ship departure from the port.
    The class also logs events and ship activities throughout the process.

    Args:
        env (simpy.Environment): The simulation environment
        chassis_bays_utilization (float): Utilization of chassis bays
        run_id (int): Unique identifier for the simulation run
        channel (Channel): The channel object for managing ship movements
        day_pilots (simpy.Container): Simpy container for day pilots available
        night_pilots (simpy.Container) : Simpy container for night pilots available
        tugboats (simpy.Container): Simpy container for tugboats available
        ship_info (dict): Information about the ships
        last_section (str): The last section of the channel
        selected_terminal (str): The terminal selected for the ship
        id (int): Unique identifier for the ship
        ship_type (str): Type of the ship (e.g., Container, Liquid, DryBulk)
        draft (float): Draft of the ship
        width (float): Width of the ship
        unload_time_per_container (float): Time taken to unload a container
        load_time_per_container (float): Time taken to load a container
        containers_to_unload (list): List of containers to be unloaded
        containers_to_load (list): List of containers to be loaded
        events (list): List to store events during the simulation
        ship_logs (list): List to store logs of the ship's activities
        port_berths (simpy.Resource): Resource representing the berths at the port
        port_yard (simpy.Store): Store representing the port's yard for containers
        SHIPS_IN_CHANNEL (int): Number of ships allowed in the channel
        SHIPS_IN_ANCHORAGE (int): Number of ships allowed in the anchorage
        terminal_data (dict): Data about the terminals, including transfer units per berth and import/export capabilities
    """
    def __init__(self, env, chassis_bays_utilization, run_id, channel, day_pilots, night_pilots, tugboats, ship_info, last_section, selected_terminal, id, ship_type, draft, width, unload_time_per_container, load_time_per_container, containers_to_unload, containers_to_load, events, ship_logs, port_berths, port_yard, SHIPS_IN_CHANNEL, SHIPS_IN_ANCHORAGE, terminal_data):
        # Initialize the environment
        self.env = env
        self.SHIPS_IN_CHANNEL = SHIPS_IN_CHANNEL
        self.SHIPS_IN_ANCHORAGE = SHIPS_IN_ANCHORAGE
        self.run_id = run_id
        self.terminal_data = terminal_data
        self.chassis_bays_utilization = chassis_bays_utilization

        # Initialize the channel
        self.channel = channel
        self.ship_info = ship_info
        self.last_section = last_section

        # day_pilots, night_pilots and tugboats
        self.day_pilots, self.night_pilots = day_pilots, night_pilots
        self.tugboats = tugboats

        # Attributes of ship
        self.id = id
        self.selected_terminal = selected_terminal
        self.ship_type = ship_type
        self.num_cranes = get_value_by_terminal(
            self.terminal_data, 'Container', selected_terminal, 'transfer units per berth')
        self.containers_to_unload = containers_to_unload
        self.containers_to_load = containers_to_load
        self.draft = draft
        self.width = width
        self.day = None

        # Attributes of container terminal
        self.unload_time = unload_time_per_container
        self.load_time = load_time_per_container
        self.time_for_uturn_min, self.time_for_uturn_max = TIME_FOR_UTURN

        # proxy for wait time
        self.cargo_wait_time = 0

        # The allocation of cranes and berth will be done during the process
        self.current_berth = None
        self.allocated_cranes = []

        # Get the ship logs and events
        self.events = events
        self.ship_logs = ship_logs

        # Allocate berth and cranes (note that cranes are part of berths)
        self.port_berths = port_berths
        self.port_yard = port_yard

        # Start the process
        self.env.process(self.process())

    def process(self):
        """
        Process for the container terminal, simulating the arrival, unloading, loading, waiting, and departure of ships.
        """

        self.ship_logs = update_ship_logs(self.ship_logs, "C", self.id, self.selected_terminal, ship_start_time=nan, time_to_get_berth=nan, time_for_restriction_in=nan, time_to_get_pilot_in=nan,
                                          time_to_get_tugs_in=nan, time_to_common_channel_in=nan, time_to_travel_channel_in=nan, time_to_tug_steer_in=nan, unloading_time=nan, loading_time=nan, waiting_time=nan,
                                          departure_time=nan, time_to_get_pilot_out=nan, time_to_get_tugs_out=nan, time_to_tug_steer_out=nan, time_for_uturn=nan,
                                          time_to_travel_channel_out=nan, time_to_common_channel_out=nan, ship_end_time=nan)

        # ship arrival
        ship_start_time = self.env.now
        update_ship_logs(self.ship_logs, "C", self.id,
                         self.selected_terminal, ship_start_time=ship_start_time)
        self.events.append((f"Ship_{self.id}", "Container", f"L.{self.selected_terminal}",
                           "arrive", self.env.now, f"Ship {self.id} arrived at the port"))

        # Anchorage waiting
        yield self.env.timeout(constants.ANCHORAGE_WAITING_CONTAINER)

        # Berth allocation
        start_time = self.env.now
        yield self.env.process(self.dock())
        time_to_get_berth = self.env.now - ship_start_time
        update_ship_logs(self.ship_logs, "C", self.id,
                         self.selected_terminal, time_to_get_berth=time_to_get_berth)

        # check day or night
        self.day = is_daytime(self.env.now, 7, 19)

        # Channel process (in)
        yield self.env.process(self.channel.channel_process(self.ship_info, self.day, self.last_section, self.id, self.selected_terminal, "C", self.run_id))

        # Terminal process
        self.events.append((f"Ship_{self.id}", "Container", f"C.{self.selected_terminal}", "dock", self.env.now,
                           f"Ship {self.id} docked to berth {self.current_berth.id} with waiting time {'N/A'}"))

        start_time = self.env.now
        yield self.env.process(self.unload())
        unloading_time = self.env.now - start_time
        update_ship_logs(self.ship_logs, "C", self.id,
                         self.selected_terminal, unloading_time=unloading_time)

        start_time = self.env.now
        yield self.env.process(self.load())
        loading_time = self.env.now - start_time
        update_ship_logs(self.ship_logs, "C", self.id,
                         self.selected_terminal, loading_time=loading_time)

        start_time = self.env.now
        yield self.env.process(self.wait())
        waiting_time = self.env.now - start_time
        update_ship_logs(self.ship_logs, "C", self.id,
                         self.selected_terminal, waiting_time=waiting_time)

        start_time = self.env.now
        yield self.env.process(self.detach_and_depart())
        departure_time = self.env.now - start_time
        update_ship_logs(self.ship_logs, "C", self.id,
                         self.selected_terminal, departure_time=departure_time)

        self.events.append((f"Ship_{self.id}", "Container", f"C.{self.selected_terminal}", "undock", self.env.now,
                           f"Ship {self.id} undocked to berth {self.current_berth.id} with waiting time {'N/A'}"))

        # Channel process (out)
        self.day = is_daytime(self.env.now, 7, 19)
        yield self.env.process(self.channel.channel_process(self.ship_info, self.day,  self.last_section, self.id, self.selected_terminal, "C", self.run_id))

        ship_end_time = self.env.now
        update_ship_logs(self.ship_logs, "C", self.id,
                         self.selected_terminal, ship_end_time=ship_end_time)
        self.events.append((f"Ship_{self.id}", "Container", f"C.{self.selected_terminal}", "depart", self.env.now,
                           f"Ship {self.id} departed from berth {self.current_berth.id} with waiting time {'N/A'}"))

    def dock(self):
        """
        Represents the process of docking the ship to a berth and allocating cranes.
        This includes waiting for a berth to become available and then getting the cranes allocated.
        """
        self.current_berth = yield self.port_berths.get()

    def unload(self):
        """
        Represents the unloading of containers from the vessel at the input crane movement rate.
        This includes requesting cranes, unloading containers, and storing them in the port yard.
        """
        import_terminal = get_value_by_terminal(
            self.terminal_data, 'Container', self.selected_terminal, 'import')
        if import_terminal:
            # Request for the required number of cranes
            for _ in range(self.num_cranes):
                crane = yield self.current_berth.cranes.get()
                self.allocated_cranes.append(crane)
            self.events.append((f"Ship_{self.id}", "Container", f"C.{self.selected_terminal}",
                               "all_cranes_attached", self.env.now, f"Ship {self.id} cranes allocated"))
            # Unload containers
            for container in self.containers_to_unload:
                unload_time_per_container = self.unload_time
                self.cargo_wait_time += unload_time_per_container / self.num_cranes
                yield self.env.timeout(unload_time_per_container / self.num_cranes)
            self.events.append((f"Ship_{self.id}", "Container", f"C.{self.selected_terminal}",
                               "all_container_unloaded", self.env.now, f"Ship {self.id} unloaded all containers"))
            # Put containers in yard
            for container in self.containers_to_unload:
                yield self.port_yard.put(container)
            self.events.append((f"Ship_{self.id}", "Container", f"C.{self.selected_terminal}",
                               "all_container_stored", self.env.now, f"Ship {self.id} unloaded all containers"))
        else:
            self.events.append((f"Ship_{self.id}", "Container", f"C.{self.selected_terminal}",
                               "all_cranes_attached", self.env.now, f"Ship {self.id} cranes allocated"))
            self.events.append((f"Ship_{self.id}", "Container", f"C.{self.selected_terminal}",
                               "all_container_unloaded", self.env.now, f"Ship {self.id} unloaded all containers"))
            self.events.append((f"Ship_{self.id}", "Container", f"C.{self.selected_terminal}",
                               "all_container_stored", self.env.now, f"Ship {self.id} unloaded all containers"))

    def load(self):
        """
        Represents the loading of containers onto the vessel at the input crane movement rate.
        This includes checking if the terminal can export containers, loading them onto the vessel, and removing them from the yard.
        """
        export_terminal = get_value_by_terminal(
            self.terminal_data, 'Container', self.selected_terminal, 'export')
        # Load containers
        if export_terminal:
            for container in self.containers_to_load:
                # Remove the container from the yard
                yield self.port_yard.get()
                # Load the containers onto the vessel
                load_time_per_container = self.load_time
                self.cargo_wait_time += load_time_per_container / self.num_cranes
                yield self.env.timeout(load_time_per_container / self.num_cranes)
            self.events.append((f"Ship_{self.id}", "Container", f"C.{self.selected_terminal}",
                               "all_container_loaded", self.env.now, f"Ship {self.id} loaded all containers"))
        else:
            self.events.append((f"Ship_{self.id}", "Container", f"C.{self.selected_terminal}",
                               "all_container_loaded", self.env.now, f"Ship {self.id} loaded all containers"))

    def wait(self):
        """
        Represents the waiting time at the port after unloading and loading operations.
        This includes calculating the waiting time based on the terminal efficiency and cargo wait time.
        """
        # Calculate the waiting time based on the terminal efficiency and cargo wait time
        port_waiting_time = (1 / CONTAINER_TERMINAL_EFFICIENCY) * \
            self.cargo_wait_time - self.cargo_wait_time
        yield self.env.timeout(port_waiting_time)

    def detach_and_depart(self):
        """
        Represents the process of detaching from the berth and departing from the port.
        This includes returning the cranes to the berth and releasing the berth for other ships.
        """
        # Return the cranes to the berth
        for crane in self.allocated_cranes:
            yield self.current_berth.cranes.put(crane)
        self.events.append((f"Ship_{self.id}", "Container", f"C.{self.selected_terminal}",
                           "all_cranes_returned", self.env.now, f"Ship {self.id} returned all cranes"))

        # Depart from the berth
        yield self.port_berths.put(self.current_berth)
