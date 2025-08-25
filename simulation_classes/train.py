"""
This module defines the Train class, which simulates the loading and unloading of cargo at a terminal.
The train processes cargo in batches, either importing or exporting based on the flags provided.
"""

from simulation_classes.port import Container

class Train(object):
    """Train class to simulate the loading and unloading of cargo at a terminal.
        Args:
            env (simpy.Environment): Simulation environment.
            train_id (str): Unique identifier for the train.
            terminal_id (str): Unique identifier for the terminal.
            car_amount (int): Number of cars in the train.
            cargo_transfer_rate (float): Rate at which cargo is transferred (e.g., pounds per hour).
            racks (simpy.Resource): Resource representing the unloading racks.
            cargo_yard (simpy.Container): Container representing the cargo yard.
            train_events (dict): Dictionary to log events related to the train.
            transfer_amount (float): Total amount of cargo to be transferred per car.
            import_bool (bool): Flag indicating if the train is importing cargo.
            export_bool (bool): Flag indicating if the train is exporting cargo.
            cargo_type (str): Type of cargo being handled ("Container" or "Bulk").
        """
    def __init__(self, env, train_id, terminal_id, car_amount, cargo_transfer_rate, racks, cargo_yard, train_events, transfer_amount, import_bool, export_bool, cargo_type):
        self.env = env
        self.train_id = train_id
        self.terminal_id = terminal_id
        self.car_amount = car_amount  
        self.cargo_transfer_rate = cargo_transfer_rate 
        self.racks = racks  # simpy.Resource object for unloading
        self.cargo_yard = cargo_yard
        self.train_events = train_events
        self.transfer_amount_per_car = transfer_amount / car_amount
        self.import_bool = import_bool
        self.export_bool = export_bool
        self.cargo_type = cargo_type
        self.env.process(self.process())

    def process(self):
        """
        Process to handle the train's loading and unloading operations.
        This method initializes the train's event log, waits for the train to arrive,
        and then either imports or exports cargo based on the flags provided.
        It also logs the time taken for each operation and the start/end times of each car.
        """
        # Initialize train event log
        self.train_events[self.train_id] = {}
        self.train_events[self.train_id]["Batch size"] = self.racks.capacity
        self.train_events[self.train_id]["Cargo type"] = self.cargo_type
        self.train_events[self.train_id]["Cargo transfer rate"] = self.cargo_transfer_rate
        self.train_events[self.train_id]["Transfer amount per car"] = self.transfer_amount_per_car
        self.train_events[self.train_id]["Import/Export"] = "Import" if self.import_bool else "Export"
        self.train_events[self.train_id]["Train ID"] = self.train_id
        self.train_events[self.train_id]["Terminal ID"] = self.terminal_id
        self.train_events[self.train_id]["Car amount"] = self.car_amount
        self.train_events[self.train_id]["Car start/end times"] = []
        self.train_events[self.train_id]["Time to get racks"] = []
        self.train_events[self.train_id]["Train Arrival"] = self.env.now    

        if self.import_bool:  # Import Process
            yield self.env.process(self.import_cargo(self.car_amount))
        if self.export_bool:  # Export Process
            yield self.env.process(self.export_cargo(self.car_amount))

        self.train_events[self.train_id]["Train Departure"] = self.env.now
        self.train_events[self.train_id]["Train Duration"] = self.train_events[self.train_id]["Train Departure"] - self.train_events[self.train_id]["Train Arrival"]

    def import_cargo(self, empty_cars):
        """
        Import cargo into the train's cars.
        This method processes the loading of cargo into the train's cars in batches.
        Args:
            empty_cars (int): Number of empty cars to be loaded with cargo.
        Yields:
            simpy.Timeout: Waits for all cars in the batch to be loaded before proceeding.
        Returns:
            None
        """
        cars_remaining = empty_cars
        while cars_remaining > 0:
            batch_size = min(cars_remaining, self.racks.capacity)
            tasks = [self.env.process(self.load_car()) for _ in range(batch_size)]
            yield self.env.all_of(tasks)  # Wait for all cars in the batch to be loaded
            cars_remaining -= batch_size

    def export_cargo(self, full_cars):
        """
        Export cargo from the train's cars.
        This method processes the unloading of cargo from the train's cars in batches.
        Args:
            full_cars (int): Number of full cars to be unloaded.
        Yields:
            simpy.Timeout: Waits for all cars in the batch to be unloaded before proceeding.
        Returns:
            None
        """
        cars_remaining = full_cars
        while cars_remaining > 0:
            batch_size = min(cars_remaining, self.racks.capacity)
            tasks = [self.env.process(self.unload_car()) for _ in range(batch_size)]
            yield self.env.all_of(tasks)  # Wait for all cars in the batch to be unloaded
            cars_remaining -= batch_size                                                                                              

    def load_car(self):
        """
        Load cargo into a single car of the train.
        This method handles the loading of cargo into a car, including waiting for the racks
        and calculating the time taken for loading.
        Yields:
            simpy.Timeout: Waits for the loading process to complete.
        Returns:
            None
        """
        car_start_time = self.env.now
        with self.racks.request() as req:
            yield req
            self.train_events[self.train_id]["Time to get racks"].append(self.env.now - car_start_time)
            load_time = self.transfer_amount_per_car / self.cargo_transfer_rate # ex: (50 lbs / car) / (20 lbs/hr) = 2.5 hr / car
            yield self.env.timeout(load_time)
            if self.cargo_type == "Container":
                num_containers =  int(self.transfer_amount_per_car)
                for i in range(num_containers):
                    yield self.cargo_yard.get()
            else:
                yield self.cargo_yard.get(self.transfer_amount_per_car)
        car_end_time = self.env.now
        self.train_events[self.train_id]["Car start/end times"].append((car_start_time, car_end_time))
            

    def unload_car(self):
        """
        Unload cargo from a single car of the train.
        This method handles the unloading of cargo from a car, including waiting for the racks
        and calculating the time taken for unloading.
        Yields:
            simpy.Timeout: Waits for the unloading process to complete.
        Returns:
            None
        """
        car_start_time = self.env.now
        with self.racks.request() as req:
            yield req
            self.train_events[self.train_id]["Time to get racks"].append(self.env.now - car_start_time)
            unload_time = self.transfer_amount_per_car / self.cargo_transfer_rate # ex: (50 lbs / car) / (20 lbs/hr) = 2.5 hr / car
            yield self.env.timeout(unload_time)
            if self.cargo_type == "Container":
                num_containers =  int(self.transfer_amount_per_car)
                for i in range(num_containers):
                    ctr = Container(id=f"{self.train_id}_unload_{i}", width=100000)
                    yield self.cargo_yard.put(ctr)
            else:
                self.cargo_yard.put(self.transfer_amount_per_car)
        car_end_time = self.env.now
        self.train_events[self.train_id]["Car start/end times"].append((car_start_time, car_end_time))

                                  
            
            
            
             
             
               
            
             
             
            
            
                                                                                                                               