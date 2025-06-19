import numpy as np
import time
import math

from visualizers.RewardVisualizer import RewardVisualizer

# Constants for rewards
from constants.constants import (
    CONTINUES_PUNISHMENT,
    INCOMPLETE_MAP_PUNISHMENT,
    HIT_WALL_PUNISHMENT,
    CLOSE_TO_WALL_PUNISHMENT,
    WALL_POWER,
    EXPLORATION_REWARD,
    MOVEMENT_REWARD,
    REVISIT_PENALTY,
    REMEMBER_VISIT_TIME,
    GO_BACK_PUNISH
)


class RewardCalculator:
    def __init__(self, linear_speed, rad_of_robot):
        self.rad_of_robot = rad_of_robot

        self.linear_speed = linear_speed
        # Initialize the reward visualizer
        print("Creating reward visualizer node...")
        self.reward_vis = RewardVisualizer(print_interval=100)  # print_interval= every x steps print slightly detailed loggs
        print("Reward visualizer node created")
        # Store recent reward components for visualization
        self.last_cont_punishment = 0
        self.last_wall_punishment = 0
        self.last_exploration_reward = 0
        self.last_movement_reward = 0
        self.last_revisit_penalty = 0
        self.last_total_reward = 0

        self.explored_threshold = 0.93  # 93%
        self.max_episode_steps = 800  # steps

        self.previous_map = None
        self.total_cells = None
        self.visit_count_map = None
        self.last_position = None
        self.map_explored_percent = 0

        self.step_counter = 0


### getters and setters

    def set_total_cells(self, total_cells):
        self.total_cells = total_cells

    def get_total_cells(self):
        return self.total_cells

### functions

    def update_visit_count(self, grid_position, new_map):
        """
        Update:
            the visit count for the current cell
            if the cell was visited more than a short time ago (a few seconds) it resets its count
        Args:
            param dt
        """
        # Initialize visit count map if it doesn't exist yet
        if self.visit_count_map is None:
            # If we have a processed map, create a matching visit count map
            if new_map:
                # Determine dimensions of the map
                crop_size = int(np.sqrt(len(new_map)))  # if map is square
                # Create 3D array to track visit counts and times
                # [0] = visit count, [1] = last visit time
                self.visit_count_map = np.zeros((crop_size, crop_size, 2), dtype=float)
                print(f"Initialized visit count map with size {crop_size}x{crop_size}")
            else:
                print("Cannot initialize visit count map - no processed map available")
                return

        # Extract grid coordinates
        _, _, grid_x, grid_y = grid_position

        # Convert to integers for array indexing
        grid_x_int = int(grid_x)
        grid_y_int = int(grid_y)

        crop_size = self.visit_count_map.shape[0]
        if 0 <= grid_x_int < crop_size and 0 <= grid_y_int < crop_size:
            # check time since last
            current_time = time.time()
            last_visit_time = self.visit_count_map[grid_x_int, grid_y_int, 1]
            time_since_last_visit = current_time - last_visit_time
            if time_since_last_visit > REMEMBER_VISIT_TIME:
                self.visit_count_map[grid_x_int, grid_y_int, 0] = 0
            self.visit_count_map[grid_x_int, grid_y_int, 1] = current_time

            # Increment visit count for this cell with a 20 visits cap
            visits = self.visit_count_map[grid_x_int, grid_y_int, 0]
            if visits < 25:
                visits += 1
                self.visit_count_map[grid_x_int, grid_y_int, 0] = visits

    def calculate_revisit_penalty(self, dt, grid_position, new_map):
        """
        Calculate penalty for revisiting cells that were visited a short time ago (a few seconds)
        Returns:
            Penalty value (negative reward)
        """
        self.update_visit_count(grid_position, new_map)

        # Check if visit map exists
        if self.visit_count_map is None:
            return 0.0

        # Extract grid coordinates
        _, _, grid_x, grid_y = grid_position

        # Convert to integers for array indexing
        grid_x_int = int(grid_x)
        grid_y_int = int(grid_y)

        # Default penalty
        penalty = 0.0

        # Ensure coordinates are within bounds
        crop_size = self.visit_count_map.shape[0]
        if 0 <= grid_x_int < crop_size and 0 <= grid_y_int < crop_size:
            # Get visit count for this cell
            visits = self.visit_count_map[grid_x_int, grid_y_int, 0]

            # Only penalize cells visited more than once
            if visits > 1:
                # Apply increasing penalty for repeated visits
                # Penalty grows with each revisit, but with diminishing returns
                penalty = REVISIT_PENALTY * visits * dt
                self.last_revisit_penalty = penalty

        return penalty

    def movement_to_reward(self, dt, odom_pos, action):
        """Calculate reward based on distance traveled since last update"""
        reward = 0

        if action == 1:  # going backwards
            reward += (GO_BACK_PUNISH * dt)

        if self.last_position is not None:
            # Calculate distance moved
            curr_pos = odom_pos  # use odom pos since it's more accurate for shorter dis (update's more frq)
            distance_moved = math.sqrt(
                (curr_pos[1] - self.last_position[1]) ** 2 +
                (curr_pos[2] - self.last_position[2]) ** 2
            )

            # Only reward significant movement (prevents micro-movements)
            if distance_moved > (self.linear_speed-0.03) * dt:  # threshold
                movement_reward = MOVEMENT_REWARD * (distance_moved * 100) * dt
                reward += movement_reward
                # Store for visualization

        # Store current position for next comparison
        self.last_position = odom_pos.copy()

        self.last_movement_reward = reward

        return reward

    def percent_explored(self, new_map):
        if self.total_cells is None or len(new_map) == 0:
            self.map_explored_percent = 0.0
            return self.map_explored_percent
        known_cells = sum(1 for val in new_map if val != -1.0)
        self.map_explored_percent = known_cells / self.total_cells
        return self.map_explored_percent

    def scale_distance_by_scan_angle(self, scan_distance, scan_idx, num_sectors=16):
        """
        Adjust the measured distance based on scan angle to make the safety distance
        more uniform around the robot despite the lidar being at the back.
        Args:
            scan_distance: Raw distance from lidar scan
            scan_idx: Index of the scan in the array (0 to num_sectors-1)
            num_sectors: Total number of lidar sectors
        Returns:
            Adjusted distance that provides consistent safety margins
        """
        # Calculate the angle in degrees for easier understanding
        robot_angle_degrees = (scan_idx * 360 / num_sectors)

        # Calculate scaling factors based on front distance (which stays at 1.0)
        front_scale = 1.0
        side_scale = 1.79  # makes side readings comparable to front
        back_scale = 2.10  # makes back readings comparable to front

        # Determine scaling factor based on the robot-relative angle
        if 330 <= robot_angle_degrees or robot_angle_degrees <= 30:  # Front section (0° ±30°)
            scale_factor = front_scale
        elif 160 <= robot_angle_degrees <= 200:  # Back section (180° ±20°)
            scale_factor = back_scale
        else:
            # Side sections
            if 30 < robot_angle_degrees < 160:  # Right side of robot
                if robot_angle_degrees < 90:  # Front-right quadrant
                    # Interpolate from front to side
                    factor = (robot_angle_degrees - 30) / 60
                    scale_factor = front_scale + (side_scale - front_scale) * factor
                else:  # Back-right quadrant
                    # Interpolate from side to back
                    factor = (robot_angle_degrees - 90) / 70
                    scale_factor = side_scale + (back_scale - side_scale) * factor
            else:  # Left side of robot (200° to 330°)
                if robot_angle_degrees < 270:  # Back-left quadrant
                    # Interpolate from back to side
                    factor = (robot_angle_degrees - 200) / 70
                    scale_factor = back_scale - (back_scale - side_scale) * factor
                else:  # Front-left quadrant
                    # Interpolate from side to front
                    factor = (robot_angle_degrees - 270) / 60
                    scale_factor = side_scale - (side_scale - front_scale) * factor

        # Actually apply the scaling factor to the distance
        adjusted_distance = scan_distance * scale_factor

        # if self.step_counter % 20 == 0:
        #     print(f"Angle {robot_angle_degrees:.1f}°: {scan_distance:.2f}m -> {adjusted_distance:.2f}m (scale: {scale_factor:.2f})")

        return adjusted_distance

    def dis_to_wall_to_punishment(self, dt, new_dis):
        """
        Calculate punishment based on adjusted distances to walls
        """
        # Apply distance adjustments based on scan angles
        adjusted_distances = []
        for idx, distance in enumerate(new_dis):
            adjusted = self.scale_distance_by_scan_angle(distance, idx, len(new_dis))
            adjusted_distances.append(adjusted)

        closest = min(adjusted_distances)
        is_terminated = False

        # Define the danger zone
        danger_zone_start = self.rad_of_robot * 2  # Start punishment from 2x radius
        danger_zone_end = self.rad_of_robot  # Max punishment at actual collision

        # Check for immediate collision
        if closest < self.rad_of_robot:
            self.last_wall_punishment = HIT_WALL_PUNISHMENT
            return HIT_WALL_PUNISHMENT, True

        # If outside danger zone, no punishment
        if closest >= danger_zone_start:
            self.last_wall_punishment = 0
            return 0, False

        # In danger zone - normalize to 0-1 range
        # How deep into the danger zone are we (0 = just entered, 1 = at collision boundary)
        danger_fraction = (danger_zone_start - closest) / (danger_zone_start - danger_zone_end)

        # When danger_fraction is small (far from wall): punishment is very small
        # When danger_fraction is near 1 (close to wall): punishment increases rapidly
        punishment_factor = danger_fraction ** WALL_POWER

        # Scale to punishment range
        punishment = punishment_factor * HIT_WALL_PUNISHMENT * CLOSE_TO_WALL_PUNISHMENT * dt

        # Clip to maximum punishment
        punishment = max(punishment, HIT_WALL_PUNISHMENT)

        # Store for visualization
        self.last_wall_punishment = punishment

        return punishment, is_terminated

    def change_in_map_to_reward(self, new_map):
        """Calculate reward based on newly discovered map cells"""
        # Skip if we don't have a previous map to compare
        if self.previous_map is None:
            self.previous_map = new_map.copy()
            self.last_exploration_reward = 0
            return 0

        # Count newly discovered cells (changed from -1 to any other value)
        reward = 0
        for i in range(len(new_map)):
            if i < len(self.previous_map):
                # Cell was unknown (-1) and is now known
                if self.previous_map[i] == -1 and new_map[i] != -1:
                    reward += EXPLORATION_REWARD  # Reward for each newly discovered cell

        # Store current map for next comparison
        self.previous_map = new_map.copy()

        # Store for visualization
        self.last_exploration_reward = reward

        return reward

    def check_steps_and_map_completion(self):
        # Check map exploration condition
        if self.map_explored_percent >= self.explored_threshold:
            print(f"Terminating: {self.map_explored_percent * 100:.2f}% of map explored (target: {self.explored_threshold * 100}%)")
            return True

        # Check max-step-based termination
        if self.max_episode_steps <= self.step_counter:
            print(f"Terminating: episode ran for {self.step_counter} steps (max: {self.max_episode_steps} steps)")
            return True

        return False

    def incomplete_map_punishment(self, dt):
        pun = 1.5 * CONTINUES_PUNISHMENT * (1.0 - self.map_explored_percent) * dt
        return pun * INCOMPLETE_MAP_PUNISHMENT

    def calc_reward(self, time_from_last_env_update, new_dis, new_map, grid_position, odom_pos, action):
        """Calculate reward based on time spent, proximity to walls, and exploration"""
        self.step_counter += 1

        self.percent_explored(new_map)

        # Time-&-map-incompletion-based continuous punishment
        cont = self.incomplete_map_punishment(time_from_last_env_update)
        cont += CONTINUES_PUNISHMENT * time_from_last_env_update / 1.5
        self.last_cont_punishment = cont
        reward = cont

        # Punishment for being close to walls
        pun, is_terminated = self.dis_to_wall_to_punishment(time_from_last_env_update, new_dis)
        reward += pun

        # Reward for exploring new areas
        exploration_reward = self.change_in_map_to_reward(new_map)
        reward += exploration_reward

        # Reward for moving to not stay in place
        movement_reward = self.movement_to_reward(time_from_last_env_update, odom_pos, action)
        reward += movement_reward

        # Add penalty for revisiting cells
        revisit_penalty = self.calculate_revisit_penalty(time_from_last_env_update, grid_position, new_map)
        reward += revisit_penalty

        # Check if finished by episode timeout or by map completion
        trunc = self.check_steps_and_map_completion()

        # Store total reward for visualization
        self.last_total_reward = reward

        # Update reward visualizer with new data
        self.reward_vis.add_reward_data(
            self.last_cont_punishment,
            self.last_wall_punishment,
            self.last_exploration_reward,
            self.last_movement_reward,
            self.last_revisit_penalty,
            self.last_total_reward
        )

        if is_terminated or trunc:
            # Stop the robot when it hits a wall or episode ends
            print("is_terminated: ", is_terminated, ", is_truncated: ", trunc)
            self.last_position = None

            return reward, True

        return reward, is_terminated

    def reward_reset(self):
        self.previous_map = None
        self.total_cells = None
        self.visit_count_map = None
        self.last_position = None
        self.map_explored_percent = 0

        self.step_counter = 0

        # Reset reward visualization
        self.reward_vis.reset_data()
