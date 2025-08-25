"""
This module defines the Channel and ChannelSection classes for simulating ship navigation through a channel.
"""
import random

from tqdm import tqdm

import constants
from simulation_analysis.results import log_line, update_ship_logs
from simulation_handler.helpers import is_daytime, save_warning

CHANNEL_SECTION_DIMENSIONS = constants.CHANNEL_SECTION_DIMENSIONS
START_DAYLIGHT_RESTRICTION_HR = constants.START_DAYLIGHT_RESTRICTION_HR
STOP_DAYLIGHT_RESTRICTION_HR = constants.STOP_DAYLIGHT_RESTRICTION_HR
TIME_FACTOR = constants.TIME_FACTOR
TIME_COMMON_CHANNEL = constants.TIME_COMMON_CHANNEL
TIME_FOR_UTURN = constants.TIME_FOR_UTURN
TIME_FOR_TUG_STEER = constants.TIME_FOR_TUG_STEER


class ChannelSection:
    """
    Class for each section of the channel    
    This class manages the dimensions and capacities of a channel section, 
    and provides methods to check if a ship can navigate through it based on its width, draft, and other restrictions.

    Args:
            env (Env): The simulation environment.
            id (int): The identifier for the channel section.
            length (float): The length of the channel section.
            width (float): The width of the channel section.
            draft (float): The draft of the channel section.
            speed (float): The speed of the channel section.
            simulation_time (int): The total simulation time.
            safeTwoway (bool): Whether the section is safe for two-way traffic.
    """

    def __init__(self, env, id, length, width, draft, speed, simulation_time, safeTwoway):
        # Initialize the channel section with its dimensions and capacities.
        self.id = id
        self.env = env
        self.width = width
        self.draft = draft
        self.length = length
        self.simulation_time = simulation_time
        self.speed = speed
        self.safeTwoway = safeTwoway

        # Initialize the containers for width and draft restrictions.
        self.width_containers = {}
        self.draft_containers = {}
        self.width_containers_in = {}
        self.draft_containers_in = {}
        self.width_containers_out = {}
        self.draft_containers_out = {}

        # Set the capacity for width and draft based on whether the section is safe for two-way traffic.
        self.width_capacity = width if not safeTwoway else 1000000
        self.draft_capacity = draft if not safeTwoway else 1000000

        self.gap_out_in = {}
        self.gap_out_out = {}

        # Initialize the containers for width and draft restrictions for each time step.
        for time_step in range(TIME_FACTOR * simulation_time):
            self.width_containers[time_step] = self.width_capacity
            self.width_containers_in[time_step] = 0
            self.width_containers_out[time_step] = 0

            self.draft_containers[time_step] = self.draft_capacity
            self.draft_containers_in[time_step] = 0
            self.draft_containers_out[time_step] = 0

            self.gap_out_in[time_step] = 1
            self.gap_out_out[time_step] = 1

    def can_accommodate_width(self, ship_info, start_time, end_time):
        """
        Check if the channel section can accommodate the width of the ship at a given time step.
        Args:
            ship_info (dict): Information about the ship, including its width and direction.
            start_time (int): The start time of the ship's passage through the section.
            end_time (int): The end time of the ship's passage through the section.
        Returns:
            bool: True if the section can accommodate the ship's width, False otherwise.
        """
        for time_step in range(max(0, int(TIME_FACTOR * start_time)-1), int(TIME_FACTOR * end_time)+1):

            if ship_info['direction'] == 'in':  # Same direction max beam restrictiom
                if ship_info['width'] <= self.width_containers_in[time_step]:
                    continue
            elif ship_info['direction'] == 'out':  # Same direction max beam restrictiom
                if ship_info['width'] <= self.width_containers_out[time_step]:
                    continue
            # opposite direction combined beam restriction
            if ship_info['direction'] == 'in':
                excess = ship_info['width'] - \
                    self.width_containers_in[time_step]
                if (self.width_containers[time_step] <= excess):
                    return False
            elif ship_info['direction'] == 'out':
                excess = ship_info['width'] - \
                    self.width_containers_out[time_step]
                if (self.width_containers[time_step] <= excess):
                    return False
        return True

    def can_accommodate_draft(self, ship_info, start_time, end_time):
        """ 
        Check if the channel section can accommodate the draft of the ship at a given time step.
        Args:
            ship_info (dict): Information about the ship, including its draft and direction.
            start_time (int): The start time of the ship's passage through the section.
            end_time (int): The end time of the ship's passage through the section.
        Returns:
            bool: True if the section can accommodate the ship's draft, False otherwise.
        """
        for time_step in range(max(0, int(TIME_FACTOR * start_time)-1), int(TIME_FACTOR * end_time)+1):

            if ship_info['direction'] == 'in':  # Same direction max draft restrictiom
                if ship_info['draft'] <= self.draft_containers_in[time_step]:
                    continue
            elif ship_info['direction'] == 'out':  # Same direction max draft restrictiom
                if ship_info['draft'] <= self.draft_containers_out[time_step]:
                    continue
            # opposite direction combined draft restriction
            if ship_info['direction'] == 'in':
                excess = ship_info['draft'] - \
                    self.draft_containers_in[time_step]
                if (self.draft_containers[time_step] <= excess):
                    return False
            elif ship_info['direction'] == 'out':
                excess = ship_info['draft'] - \
                    self.draft_containers_out[time_step]
                if (self.draft_containers[time_step] <= excess):
                    return False
        return True

    def can_accomodate_gap_out(self, start_time, ship_info):
        """
        Check if the channel section can accommodate the minimum spacing for the ship at a given time step.
        Args:
            start_time (int): The start time of the ship's passage through the section.
            ship_info (dict): Information about the ship, including its direction.
        Returns:
            bool: True if the section can accommodate the minimum spacing, False otherwise.
        """

        if ship_info['direction'] == 'in':
            for time_step in range(int(TIME_FACTOR * start_time)-1, int(TIME_FACTOR * (start_time + constants.CHANNEL_ENTRY_SPACING))+1):
                if self.gap_out_in[time_step] == 1:
                    continue
                return False
            return True
        else:
            for time_step in range(int(TIME_FACTOR * start_time)-1, int(TIME_FACTOR * (start_time + constants.CHANNEL_ENTRY_SPACING))+1):
                if self.gap_out_out[time_step] == 1:
                    continue
                return False
            return True

    def update_usage(self, run_id, ship_info, start_time, end_time):
        """
        Update the usage of the channel section based on the ship's information.
        Args:
            run_id (int): The identifier for the current simulation run.
            ship_info (dict): Information about the ship, including its width, draft, and direction.
            start_time (int): The start time of the ship's passage through the section.
            end_time (int): The end time of the ship's passage through the section.
        Returns:
            None
        """
        for time_step in range(int(TIME_FACTOR * start_time), int(TIME_FACTOR * end_time)):

            if ship_info['direction'] == 'in':
                if ship_info['width'] > self.width_containers_in[time_step]:
                    excess_width_in = ship_info['width'] - \
                        self.width_containers_in[time_step]
                    self.width_containers_in[time_step] += excess_width_in
                    self.width_containers[time_step] -= excess_width_in

                if ship_info['draft'] > self.draft_containers_in[time_step]:
                    excess_draft_in = ship_info['draft'] - \
                        self.draft_containers_in[time_step]
                    self.draft_containers_in[time_step] += excess_draft_in
                    self.draft_containers[time_step] -= excess_draft_in

            elif ship_info['direction'] == 'out':
                if ship_info['width'] > self.width_containers_out[time_step]:
                    excess_width_out = ship_info['width'] - \
                        self.width_containers_out[time_step]
                    self.width_containers_out[time_step] += excess_width_out
                    self.width_containers[time_step] -= excess_width_out

                if ship_info['draft'] > self.draft_containers_out[time_step]:
                    excess_draft_out = ship_info['draft'] - \
                        self.draft_containers_out[time_step]
                    self.draft_containers_out[time_step] += excess_draft_out
                    self.draft_containers[time_step] -= excess_draft_out

            # assert that the total width and draft in the section is equal to the capacity
            assert abs(self.width_containers[time_step] + self.width_containers_in[time_step] +
                       self.width_containers_out[time_step] - self.width_capacity) <= 0.1
            assert abs(self.draft_containers[time_step] + self.draft_containers_in[time_step] +
                       self.draft_containers_out[time_step] - self.draft_capacity) <= 0.1
            # assert that the width and draft containers are non-negative
            assert self.width_containers[time_step] >= 0
            assert self.draft_containers[time_step] >= 0

    def space_out(self, start_time, ship_info):
        """ 
        Space out the channel section for the ship based on its direction and start time.
        Args:
            start_time (int): The start time of the ship's passage through the section.
            ship_info (dict): Information about the ship, including its direction.
        Returns:
            None
        """
        if ship_info['direction'] == 'in':
            for time_step in range(int(TIME_FACTOR * start_time), int(TIME_FACTOR * (start_time + constants.CHANNEL_ENTRY_SPACING))):
                assert self.gap_out_in[time_step] == 1
                self.gap_out_in[time_step] = 0
        else:
            for time_step in range(int(TIME_FACTOR * start_time), int(TIME_FACTOR * (start_time + constants.CHANNEL_ENTRY_SPACING))):
                assert self.gap_out_out[time_step] == 1
                self.gap_out_out[time_step] = 0


class Channel:
    """
    Class for the entire channel.

    This class manages the sections of the channel and handles ship navigation through it. It implements the necessary methods to check for restrictions, schedule ships, and log their movements.

    Args:
        ship_logs (list): List to log ship information.
        env (Env): The simulation environment.
        numSections (int): Number of sections in the channel.
        simulation_time (int): Total simulation time for the channel.
        safeTwoway (bool): Whether the channel is safe for two-way traffic.
        channel_events (list): List to log channel events.
        channel_logs (list): List to log channel information.
        day_pilots (simpy.Container): Simpy container for day pilots.
        night_pilots (simpy.Container): Simpy container for night pilots.
        tugboats (simpy.Container): Simpy container for tugboats.
        turnoffTime (dict): Dictionary with turnoff times and conditions.
        channel_scheduer (Resource): Resource for scheduling channel usage.
        seed (int): Random seed for reproducibility.
"""


    def __init__(self, ship_logs, env, numSections, simulation_time, safeTwoway, channel_events, channel_logs, day_pilots, night_pilots, tugboats, turnoffTime, channel_scheduer, seed):
        self.seed = seed
        self.env = env
        self.ship_logs = ship_logs
        self.numSections = numSections
        self.safeTwoway = safeTwoway
        self.channel_events = channel_events
        self.channel_logs = channel_logs
        self.simulation_time = simulation_time
        self.day_pilots = day_pilots
        self.night_pilots = night_pilots
        self.tugboats = tugboats
        self.turnoffTime = turnoffTime
        self.channel_scheduer = channel_scheduer

        self.sections = [ChannelSection(env, i, *CHANNEL_SECTION_DIMENSIONS[i], int(1.1*self.simulation_time),
                                        self.safeTwoway) for i in tqdm(range(1, numSections + 1),  desc="Creating channel")]
        random.seed(seed)

    def channel_closed(self, time):
        """
        Check if the channel is closed at a given time step.
        Args:
            time (int): The time step to check for channel closure.
        """
        if self.turnoffTime["switch"] == "channel_closed":
            for start_end_tuples in self.turnoffTime["closed_between"]:
                if start_end_tuples[0] <= time < start_end_tuples[1]:
                    return True
        return False

    def channel_closed_restriction(self, ship_info, lastSection, enter_time):
        """
        Check if the ship has beam restriction in the channel in any section at a given time step.
        Logic: 
        - If the ship is entering the channel, check each section from the start to the last section.
        - If the ship is leaving the channel, check each section from the last section to the start.
        - For each section, calculate the end time based on the ship's speed and length of the section.
        - For each section, check if the channel is closed at the end time.
        - If the channel is closed at the end time, return True (indicating the channel is closed for the ship).
        - If no section is closed, return False (indicating the channel is open for the ship).
        Args:
            ship_info (dict): Information about the ship, including its direction.
            lastSection (int): The last section of the channel to check.
            enter_time (int): The time step when the ship enters the channel.
        Returns:
            bool: True if the channel is closed for the ship, False otherwise.
        """
        if ship_info['direction'] == 'in':
            start_time = enter_time
            for i, section in enumerate(self.sections):
                if i > lastSection:
                    break
                speed = section.speed
                end_time = start_time + section.length / speed
                if self.channel_closed(end_time):
                    return True
                start_time = end_time
            return False
        elif ship_info['direction'] == 'out':
            start_time = enter_time
            for i, section in enumerate(reversed(self.sections)):
                section_number = len(self.sections) - i - 1
                if section_number > lastSection:
                    continue
                speed = section.speed
                end_time = start_time + section.length / speed
                if self.channel_closed(end_time):
                    return True
                start_time = end_time
            return False

    def beam_restriction(self, ship_info, lastSection, enter_time):
        """
        Check if the ship has beam restriction in the channel in any section at a given time step.
        Logic: 
        - If the ship is entering the channel, check each section from the start to the last section.
        - If the ship is leaving the channel, check each section from the last section to the start.
        - For each section, calculate the end time based on the ship's speed and length of the section.
        - For each section, check if the width of the ship exceeds the available width in the section.
        - If the width of the ship exceeds the available width in any section, return True (indicating beam restriction).
        - If no section has beam restriction, return False (indicating no beam restriction).
        Args:
            ship_info (dict): Information about the ship, including its width and direction.
            lastSection (int): The last section of the channel to check.
            enter_time (int): The time step when the ship enters the channel.
        Returns:
            bool: True if the ship has beam restriction, False otherwise.
        """
        if ship_info['direction'] == 'in':
            start_time = enter_time
            for i, section in enumerate(self.sections):
                if i > lastSection:
                    break
                speed = section.speed
                end_time = start_time + section.length / speed
                if not section.can_accommodate_width(ship_info, start_time, end_time):
                    return True
                start_time = end_time
            return False
        elif ship_info['direction'] == 'out':
            start_time = enter_time
            for i, section in enumerate(reversed(self.sections)):
                section_number = len(self.sections) - i - 1
                if section_number > lastSection:
                    continue
                speed = section.speed
                end_time = start_time + section.length / speed
                if not section.can_accommodate_width(ship_info, start_time, end_time):
                    return True
                start_time = end_time
            return False

    def draft_restriction(self, ship_info, lastSection, enter_time):
        """
        Check if the ship has draft restriction in the channel in any section at a given time step.
        Logic is similar to beam_restriction, but checks for draft instead of width.
        Args:
            ship_info (dict): Information about the ship, including its draft and direction.
            lastSection (int): The last section of the channel to check.
            enter_time (int): The time step when the ship enters the channel.
        Returns:
            bool: True if the ship has draft restriction, False otherwise.
        """
        if ship_info['direction'] == 'in':
            start_time = enter_time
            for i, section in enumerate(self.sections):
                if i > lastSection:
                    break
                speed = section.speed
                end_time = start_time + section.length / speed
                if not section.can_accommodate_draft(ship_info, start_time, end_time):
                    return True
                start_time = end_time
            return False
        elif ship_info['direction'] == 'out':
            start_time = enter_time
            for i, section in enumerate(reversed(self.sections)):
                section_number = len(self.sections) - i - 1
                if section_number > lastSection:
                    continue
                speed = section.speed
                end_time = start_time + section.length / speed
                if not section.can_accommodate_draft(ship_info, start_time, end_time):
                    return True
                start_time = end_time
            return False

    def minimal_spacing_not_avlbl(self, ship_info, lastSection, enter_time):
        """
        Check if the minimal spacing for the ship is not available in the channel in any section at a given time step.
        Logic is similar to beam_restriction, but checks for minimal spacing instead of width.
        Args:
            ship_info (dict): Information about the ship, including its direction.
            lastSection (int): The last section of the channel to check.
            enter_time (int): The time step when the ship enters the channel.
        Returns:
            bool: True if the minimal spacing is not available, False otherwise.
        """
        if ship_info['direction'] == 'in':
            start_time = enter_time
            for i, section in enumerate(self.sections):
                if i > lastSection:
                    break
                speed = section.speed
                end_time = start_time + section.length / speed
                if not section.can_accomodate_gap_out(start_time, ship_info):
                    return True
                start_time = end_time
            return False
        elif ship_info['direction'] == 'out':
            start_time = enter_time
            for i, section in enumerate(reversed(self.sections)):
                section_number = len(self.sections) - i - 1
                if section_number > lastSection:
                    continue
                speed = section.speed
                end_time = start_time + section.length / speed
                if not section.can_accomodate_gap_out(start_time, ship_info):
                    return True
                start_time = end_time
            return False

    def daylight_restriction(self, ship_info, lastSection, enter_time):
        """
        Check if the ship has daylight restriction in the channel at a given time step.
        Logic:
        - Check the ship's type and dimensions (width and length).
        - If the ship is a Container, Liquid, or DryBulk type, check if its width or length exceeds the specified limits.
        - If the ship exceeds the limits, check if it is daytime at the time of entry.
        - If it is not daytime, return True (indicating daylight restriction).
        - If the ship does not exceed the limits or it is daytime, return False (indicating no daylight restriction).
        Args:
            ship_info (dict): Information about the ship, including its type, width, and length.
            lastSection (int): The last section of the channel to check.
            enter_time (int): The time step when the ship enters the channel.
        Returns:
            bool: True if the ship has daylight restriction, False otherwise.
        """
        if ship_info['ship_type'] == 'Container':
            if ship_info['width'] > 120 or ship_info['length'] > 1000:
                # if not daytime return True
                return not is_daytime(enter_time, START_DAYLIGHT_RESTRICTION_HR, STOP_DAYLIGHT_RESTRICTION_HR)
        elif ship_info['ship_type'] == 'Liquid':
            if ship_info['width'] > 120 or ship_info['length'] > 900:
                return not is_daytime(enter_time, START_DAYLIGHT_RESTRICTION_HR, STOP_DAYLIGHT_RESTRICTION_HR)
        elif ship_info['ship_type'] == 'DryBulk':
            if ship_info['width'] > 120 or ship_info['length'] > 900:
                return not is_daytime(enter_time, START_DAYLIGHT_RESTRICTION_HR, STOP_DAYLIGHT_RESTRICTION_HR)
        return False

    def scheduler(self, ship_info, lastSection, run_id, enter_time):
        """
        Schedule the ship to navigate through the channel. This happens before the ship actually moves through the channel.
        Logic:
        - Check if the ship has any restrictions in the channel (beam, draft, daylight, minimal spacing).
        - If there are any restrictions, log the ship's information and return.
        - If there are no restrictions, update the ship's logs and schedule the ship to move through the channel.
        - If the ship is entering the channel, update the usage of each section in the channel.
        - If the ship is leaving the channel, update the usage of each section in reverse order.
        - Log the ship's movement through the channel.
        - Finally, yield the ship's movement through the channel.
        Note this is different from the move_through_channel method, which is called when the ship actually moves through the channel.
        The scheduler method is called as soon as the ship requests to enter the channel.
        If the ship has any restrictions, it will not be scheduled to move through the channel.
        Args:
            ship_info (dict): Information about the ship, including its direction, width, draft, and speed.
            lastSection (int): The last section of the channel to check.
            run_id (int): The identifier for the current simulation run.
            enter_time (int): The time step when the ship enters the channel.
        Returns:
            None
        """
        start_time = enter_time

        if ship_info['direction'] == 'in':
            for i, section in enumerate(self.sections):
                if i > lastSection:
                    break
                speed = section.speed
                end_time = start_time + section.length / speed
                section.update_usage(run_id, ship_info, start_time, end_time)
                self.sections[i].space_out(start_time, ship_info)
                start_time = end_time

        elif ship_info['direction'] == 'out':
            for i, section in enumerate(reversed(self.sections)):
                section_number = len(self.sections) - i - 1
                if section_number > lastSection:
                    continue
                speed = section.speed
                end_time = start_time + section.length / speed
                section.update_usage(run_id, ship_info, start_time, end_time)
                self.sections[section_number].space_out(start_time, ship_info)
                start_time = end_time

    def move_through_channel(self, ship_info, lastSection, run_id):
        """
        Simulate the movement of the ship through the channel.
        Logic:
        - If the ship is entering the channel, iterate through each section from the start to the last section.
        - If the ship is leaving the channel, iterate through each section from the last section to the start.
        - For each section, calculate the speed and end time based on the ship's speed and length of the section.
        - Log the ship's movement through the section, including its ship_id, ship_type, width, draft, and direction.
        - Yield the ship's movement through the section.
        - Log the time spent in each section.
        - Finally, update the ship's logs with the time spent in the channel.
        - If the ship is entering the channel, log the time spent in each section going in.
        - If the ship is leaving the channel, log the time spent in each section going out.
        Note this is different from the scheduler method,which schedules before the ship actually moves through the channel.
        The sheduler happens as soon as the ship requests to enter the channel.
        The move_through_channel method is called when the ship actually moves through the channel (when there are no restrictions).
        Args:
            ship_info (dict): Information about the ship, including its ship_id, ship_type, width, draft, and direction.
            lastSection (int): The last section of the channel to check.
            run_id (int): The identifier for the current simulation run.
        Returns:
            None
        """
        enter_time = self.env.now
        start_time = enter_time

        if ship_info['direction'] == 'in':
            for i, section in enumerate(self.sections):
                if i > lastSection:
                    break
                speed = section.speed
                end_time = start_time + section.length / speed
                yield self.env.timeout(section.length / speed)
                log_line(run_id, "ships_in_channel.txt",
                         f"Ship {ship_info['ship_id']} of type {ship_info['ship_type']} of width {ship_info['width']} and draft {ship_info['draft']} spent from {start_time} to {end_time} in section {i} going in")
                start_time = end_time

        elif ship_info['direction'] == 'out':
            for i, section in enumerate(reversed(self.sections)):
                section_number = len(self.sections) - i - 1
                if section_number > lastSection:
                    continue
                speed = section.speed
                end_time = start_time + section.length / speed
                yield self.env.timeout(section.length / speed)
                log_line(run_id, "ships_in_channel.txt",
                         f"Ship {ship_info['ship_id']} of type {ship_info['ship_type']} of width {ship_info['width']} and draft {ship_info['draft']} spent from {start_time} to {end_time} in section {section_number} going out")
                start_time = end_time

    def check_restrictions(self, ship_info, lastSection, request_time, wait_beam, wait_draft, wait_daylight):
        """
        Check if the ship has any restrictions in the channel at a given time step.
        Args:
            ship_info (dict): Information about the ship, including its direction, width, draft, and ship type.
            lastSection (int): The last section of the channel to check.
            request_time (int): The time step when the ship requests to enter the channel.
            wait_beam (int): The number of time steps the ship has waited for beam restrictions.
            wait_draft (int): The number of time steps the ship has waited for draft restrictions.
            wait_daylight (int): The number of time steps the ship has waited for daylight restrictions.
        Returns:
            tuple: A tuple containing:
                - restrictions (bool): True if there are any restrictions, False otherwise.
                - wait_beam (int): The updated number of time steps the ship has waited for beam restrictions.
                - wait_draft (int): The updated number of time steps the ship has waited for draft restrictions.
                - wait_daylight (int): The updated number of time steps the ship has waited for daylight restrictions.
        """
        channel_closed_restriction = self.channel_closed_restriction(
            ship_info, lastSection, request_time)
        beam_restriction = self.beam_restriction(
            ship_info, lastSection, request_time)
        draft_restriction = self.draft_restriction(
            ship_info, lastSection, request_time)
        daylight_restriction = self.daylight_restriction(
            ship_info, lastSection, request_time)

        if beam_restriction:
            wait_beam += 1
        if draft_restriction:
            wait_draft += 1
        if daylight_restriction:
            wait_daylight += 1

        restrictions = beam_restriction or draft_restriction or daylight_restriction or channel_closed_restriction

        return restrictions, wait_beam, wait_draft, wait_daylight

    def channel_process(self, ship_info, day, lastSection, ship_id, ship_terminal, ship_type, run_id):
        """
        Process for handling the ship's entry into the channel.
        It handles the scheduling and movement of the ship through the channel, including checking for restrictions and logging the ship's information.
        The process has the following steps:
        1. Check if the ship is entering or leaving the channel based on its direction.
        2. If the ship is entering the channel, then:
            2.1. Get the pilots based on the time of day (day or night).
            2.2. Check when the ship is allowed to enter being free of restrictions.
            2.3 Schedule the ship to move through the channel.
            2.4 Wait till the ship can move through the channel.
            2.5 Move the ship through the channel.
            2.6 Get the tugboats and steer the ship.
        3. If the ship is leaving the channel, then:
            3.1. Get the pilots based on the time of day (day or night).
            3.2 Get the tugboats and steer the ship.
            3.3.Check when the ship is allowed to leave being free of restrictions.
            3.4 Schedule the ship to move through the channel.
            3.5 Wait till the ship can move through the channel.
            3.6 Move the ship through the channel.
        Args:
            ship_info (dict): Information about the ship, including its direction, pilots, tugboats, and other attributes.
            day (bool): Whether it is daytime or nighttime.
            lastSection (int): The last section of the channel to check.
            ship_id (int): The identifier for the ship.
            ship_terminal (str): The terminal where the ship is headed.
            ship_type (str): The type of the ship (e.g., Container, Liquid, DryBulk).
            run_id (int): The identifier for the current simulation run.
        Returns:
            None
        """
        lastSection = lastSection - 1
        request_time_initial = self.env.now

        inbound_closed_tuples = constants.INBOUND_CLOSED
        outbound_closed_tuples = constants.OUTBOUND_CLOSED

        wait_queue = self.env.now - request_time_initial

        # If ship travels in
        if ship_info['direction'] == 'in':
            ship_entering_channel_time = self.env.now

            # get pilots
            day = is_daytime(ship_entering_channel_time, 7, 19)

            if day:
                yield self.day_pilots.get(ship_info['pilots'])
            else:
                yield self.night_pilots.get(ship_info['pilots'])
            time_to_get_pilot = self.env.now - ship_entering_channel_time
            update_ship_logs(self.ship_logs, ship_type, ship_id,
                             ship_terminal, time_to_get_pilot_in=time_to_get_pilot)

            # travel anchorage to mouth of the channel
            time_to_common_channel = random.uniform(
                TIME_COMMON_CHANNEL[0], TIME_COMMON_CHANNEL[1])
            yield self.env.timeout(time_to_common_channel)
            time_to_common_channel = self.env.now - \
                (ship_entering_channel_time + time_to_get_pilot)
            update_ship_logs(self.ship_logs, ship_type, ship_id, ship_terminal,
                             time_to_common_channel_in=time_to_common_channel)

            # Schedule the ship to move through the channel
            with self.channel_scheduer.request() as request:
                yield request

                request_time = self.env.now
                wait_daylight = 0
                wait_beam = 0
                wait_draft = 0
                # Note that multiple restrictions can be active at the same time
                wait_total_restriction = 0
                wait_total = 0

                if not self.safeTwoway:
                    restrcictions, wait_beam, wait_draft, wait_daylight = self.check_restrictions(
                        ship_info, lastSection, request_time, wait_beam, wait_draft, wait_daylight)
                    channel_closwed = self.channel_closed(request_time)
                    channel_closed_inbound = False

                    for start, end in inbound_closed_tuples:
                        if start <= request_time <= end:
                            channel_closed_inbound = True

                    no_spacing = self.minimal_spacing_not_avlbl(
                        ship_info, lastSection, request_time)

                    while restrcictions or channel_closwed or channel_closed_inbound or no_spacing:
                        if channel_closed_inbound:
                            print("Channel closed inbound at time", request_time)

                        wait_total += 1
                        wait_total_restriction += 1
                        request_time += 1
                        restrcictions, wait_beam, wait_draft, wait_daylight = self.check_restrictions(
                            ship_info, lastSection, request_time, wait_beam, wait_draft, wait_daylight)
                        channel_closwed = self.channel_closed(request_time)
                        no_spacing = self.minimal_spacing_not_avlbl(
                            ship_info, lastSection, request_time)
                        channel_closed_inbound = False
                        for start, end in inbound_closed_tuples:
                            if start <= request_time <= end:
                                channel_closed_inbound = True

                else:
                    channel_closwed = self.channel_closed(request_time)

                    channel_closed_inbound = False
                    for start, end in inbound_closed_tuples:
                        if start <= request_time <= end:
                            channel_closed_inbound = True

                    no_spacing = self.minimal_spacing_not_avlbl(
                        ship_info, lastSection, request_time)

                    while channel_closwed or channel_closed_inbound or no_spacing:
                        wait_total += 1
                        request_time += 1
                        channel_closwed = self.channel_closed(request_time)
                        no_spacing = self.minimal_spacing_not_avlbl(
                            ship_info, lastSection, request_time)
                        channel_closed_inbound = False
                        for start, end in inbound_closed_tuples:
                            if start <= request_time <= end:
                                channel_closed_inbound = True

                update_ship_logs(self.ship_logs, ship_type, ship_id, ship_terminal,
                                 time_for_restriction_in=f'B:{wait_beam}, D:{wait_draft}, DL:{wait_daylight}, Q:{wait_queue}, T:{wait_total}')

                self.scheduler(ship_info, lastSection, run_id,
                               enter_time=request_time)

            # travel through the channel
            yield self.env.timeout(request_time - self.env.now)
            time_entered_channel = self.env.now
            yield self.env.process(self.move_through_channel(ship_info, lastSection, run_id))
            time_to_travel_channel = self.env.now - time_entered_channel
            update_ship_logs(self.ship_logs, ship_type, ship_id, ship_terminal,
                             time_to_travel_channel_in=time_to_travel_channel)

            # get tugs
            time_tug_req = self.env.now
            yield self.tugboats.get(ship_info['tugboats'])
            time_to_get_tugs = self.env.now - time_tug_req
            update_ship_logs(self.ship_logs, ship_type, ship_id,
                             ship_terminal, time_to_get_tugs_in=time_to_get_tugs)

            # steer the ship
            time_to_tug_steer = random.uniform(
                TIME_FOR_TUG_STEER[0], TIME_FOR_TUG_STEER[1])
            yield self.env.timeout(time_to_tug_steer)
            update_ship_logs(self.ship_logs, ship_type, ship_id,
                             ship_terminal, time_to_tug_steer_in=time_to_tug_steer)

            # put the tugs back
            yield self.tugboats.put(ship_info['tugboats'])

            # Release pilots
            if day:
                yield self.day_pilots.put(ship_info['pilots'])
            else:
                yield self.night_pilots.put(ship_info['pilots'])

            ship_info['direction'] = 'out' if ship_info['direction'] == 'in' else 'in'

        # If ship travels out
        # Dont enter this if ship is going in even though ship_info['direction'] is 'out'
        elif ship_info['direction'] == 'out':
            ship_entering_channel_time = self.env.now

            # get pilots
            day = is_daytime(ship_entering_channel_time, 7, 19)
            if day:
                yield self.day_pilots.get(ship_info['pilots'])
            else:
                yield self.night_pilots.get(ship_info['pilots'])
            time_to_get_pilot = self.env.now - ship_entering_channel_time
            update_ship_logs(self.ship_logs, ship_type, ship_id,
                             ship_terminal, time_to_get_pilot_out=time_to_get_pilot)

            # get tugs
            yield self.tugboats.get(ship_info['tugboats'])
            time_to_get_tugs = self.env.now - \
                (ship_entering_channel_time + time_to_get_pilot)
            update_ship_logs(self.ship_logs, ship_type, ship_id,
                             ship_terminal, time_to_get_tugs_out=time_to_get_tugs)

            # make a uturn
            time_to_uturn = random.uniform(
                TIME_FOR_UTURN[0], TIME_FOR_UTURN[1])
            yield self.env.timeout(time_to_uturn)
            time_for_uturn = self.env.now - \
                (ship_entering_channel_time + time_to_get_pilot + time_to_get_tugs)
            update_ship_logs(self.ship_logs, ship_type, ship_id,
                             ship_terminal, time_for_uturn=time_for_uturn)

            # steer the ship
            time_to_tug_steer = random.uniform(
                TIME_FOR_TUG_STEER[0], TIME_FOR_TUG_STEER[1])
            yield self.env.timeout(time_to_tug_steer)
            update_ship_logs(self.ship_logs, ship_type, ship_id,
                             ship_terminal, time_to_tug_steer_out=time_to_tug_steer)

            # put the tugs back
            yield self.tugboats.put(ship_info['tugboats'])

            # Schedule the ship to move through the channel
            with self.channel_scheduer.request() as request:
                yield request
                request_time = self.env.now
                wait_daylight = 0
                wait_beam = 0
                wait_draft = 0
                # Note that multiple restrictions can be active at the same time
                wait_total_restriction = 0
                wait_queue = request_time - request_time_initial
                wait_total = 0
                if not self.safeTwoway:
                    restrcictions, wait_beam, wait_draft, wait_daylight = self.check_restrictions(
                        ship_info, lastSection, request_time, wait_beam, wait_draft, wait_daylight)
                    channel_closwed = self.channel_closed(request_time)
                    channel_closed_outbound = False
                    for start, end in outbound_closed_tuples:
                        if start <= request_time <= end:
                            channel_closed_outbound = True
                    no_spacing = self.minimal_spacing_not_avlbl(
                        ship_info, lastSection, request_time)
                    while restrcictions or channel_closed_outbound or channel_closwed or no_spacing:

                        if channel_closed_outbound:
                            print("Channel closed outbound at time", request_time)

                        wait_total += 1
                        wait_total_restriction += 1
                        request_time += 1
                        restrcictions, wait_beam, wait_draft, wait_daylight = self.check_restrictions(
                            ship_info, lastSection, request_time, wait_beam, wait_draft, wait_daylight)
                        channel_closwed = self.channel_closed(request_time)
                        no_spacing = self.minimal_spacing_not_avlbl(
                            ship_info, lastSection, request_time)

                        channel_closed_outbound = False
                        for start, end in outbound_closed_tuples:
                            if start <= request_time <= end:
                                channel_closed_outbound = True

                else:
                    channel_closwed = self.channel_closed(request_time)
                    channel_closed_outbound = False

                    for start, end in outbound_closed_tuples:
                        if start <= request_time <= end:
                            channel_closed_outbound = True
                    no_spacing = self.minimal_spacing_not_avlbl(
                        ship_info, lastSection, request_time)
                    no_spacing = self.minimal_spacing_not_avlbl(
                        ship_info, lastSection, request_time)
                    while channel_closwed or channel_closed_outbound or no_spacing:
                        wait_total += 1
                        request_time += 1
                        channel_closwed = self.channel_closed(request_time)
                        no_spacing = self.minimal_spacing_not_avlbl(
                            ship_info, lastSection, request_time)

                        channel_closed_outbound = False
                        for start, end in outbound_closed_tuples:
                            if start <= request_time <= end:
                                channel_closed_outbound = True

                update_ship_logs(self.ship_logs, ship_type, ship_id, ship_terminal,
                                 time_for_restriction_out=f'B:{wait_beam}, D:{wait_draft}, DL:{wait_daylight}, Q:{wait_queue}, T:{wait_total}')
                self.scheduler(ship_info, lastSection, run_id,
                               enter_time=request_time)

            # travel through the channel
            yield self.env.timeout(request_time - self.env.now)
            time_entered_channel = self.env.now
            yield self.env.process(self.move_through_channel(ship_info, lastSection, run_id))
            time_to_travel_channel = self.env.now - time_entered_channel
            update_ship_logs(self.ship_logs, ship_type, ship_id, ship_terminal,
                             time_to_travel_channel_out=time_to_travel_channel)

            # travel from mouth of the channel to anchorage
            time_to_common_channel = random.uniform(
                TIME_COMMON_CHANNEL[0], TIME_COMMON_CHANNEL[1])
            yield self.env.timeout(time_to_common_channel)
            update_ship_logs(self.ship_logs, ship_type, ship_id, ship_terminal,
                             time_to_common_channel_out=time_to_common_channel)

            # Release pilots
            if day:
                yield self.day_pilots.put(ship_info['pilots'])
            else:
                yield self.night_pilots.put(ship_info['pilots'])
