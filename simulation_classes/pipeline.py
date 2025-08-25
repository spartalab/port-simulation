"""
This module defines the Pipeline class, which simulates the flow of liquid in a pipeline system.
These pipelines can operate in two modes: 'sink' for draining liquid from a storage tank and 'source' for filling it.
This pipelines only connect storage tanks and not tanker vessels.
"""

import constants


class Pipeline:
    """
    Pipeline class to simulate the flow of liquid in a pipeline system.
    Args:
        run_id (int): Unique identifier for the simulation run.
        env (simpy.Environment): Simulation environment.
        tank (simpy.Container): Container representing the storage tank.    
        mode (str): Mode of operation, either 'sink' or 'source'.
        rate (float): Rate at which the pipeline operates, in units per time step.
    """

    def __init__(self, run_id, env, tank, mode, rate):

        self.env = env
        self.run_id = run_id
        self.capacity = tank.capacity
        self.storage = tank
        self.mode = mode
        self.rate = rate
        self.process = env.process(self.run())

    def run(self):
        """
        Simulate pipeline operation.
        Logic is as follows --
        If mode is 'sink', empty the storage at the rate of 'rate' units per time step unless the storage is empty.
        If mode is 'source', fill the storage at the rate of 'rate' units per time step unless the storage is full.
        Args:
            None
        Yields:
            simpy.Timeout: Waits for 1 time unit before the next operation.
        Raises:
            ValueError: If the rate exceeds the storage level or capacity limits.
        """
        if self.mode == "sink" and constants.LIQ_PIPELINE_OVERRIDE == True:
            while self.storage.level > 0.7 * self.capacity:

                # ensure rate is not greater than the storage level
                if self.storage.level <= self.rate:
                    raise ValueError(
                        f"SINK: Rate cannot be greater than the  0.2 * self.capacity; rate: {self.rate}, storage level: {self.storage.level}, capacity: {self.capacity}")

                yield self.storage.get(self.rate)
                yield self.env.timeout(1)

        elif self.mode == "source" and constants.LIQ_PIPELINE_OVERRIDE == True:

            # ensure rate is not greater than the storage level
            if self.storage.level + self.rate >= self.capacity:
                raise ValueError(
                    f"SOURCE: Rate cannot be greater than the  0.2 * self.capacity; rate: {self.rate}, storage level: {self.storage.level}, capacity: {self.capacity}")

            while self.storage.level <= 0.3 * self.capacity:
                yield self.storage.put(self.rate)
                yield self.env.timeout(1)
