import rclpy
from rclpy.node import Node
from collections import deque


class RewardVisualizer(Node):
    """
    Lightweight ROS2 Node for tracking reward components
    Corrected to properly reset and calculate totals
    """

    def __init__(self, print_interval=10):
        super().__init__('reward_visualizer')

        # Save parameters
        self.print_interval = print_interval
        self.step_counter = 0

        # Track episode number
        self.episode_number = 0

        # Track only cumulative values and recent averages
        self.cum_continuous_punishment = 0
        self.cum_wall_punishment = 0
        self.cum_exploration_reward = 0
        self.cum_movement_reward = 0
        self.cum_revisit_penalty = 0
        self.cum_total_reward = 0

        # Recent values for calculating averages (store the actual values, not running sums)
        self.recent_values = {
            'continuous': deque(maxlen=self.print_interval),
            'wall': deque(maxlen=self.print_interval),
            'exploration': deque(maxlen=self.print_interval),
            'movement': deque(maxlen=self.print_interval),
            'revisit': deque(maxlen=self.print_interval),
            'total': deque(maxlen=self.print_interval)
        }

        # Track min/max for each component
        self.min_values = {'continuous': 0, 'wall': 0, 'exploration': 0, 'movement': 0, 'revisit': 0, 'total': 0}
        self.max_values = {'continuous': 0, 'wall': 0, 'exploration': 0, 'movement': 0, 'revisit': 0, 'total': 0}

        self.get_logger().info("Reward tracker initialized with revisit penalty tracking")

    def add_reward_data(self, continuous, wall, exploration, movement, revisit, total=None):
        """
        Add a new data point with minimal processing
        Ensure the total is correctly calculated if needed
        """
        # Calculate correct total if needed (in case the provided total doesn't match)
        calculated_total = continuous + wall + exploration + movement + revisit

        # If total is provided, use it; otherwise use calculated total
        actual_total = total if total is not None else calculated_total

        # Update cumulative values
        self.cum_continuous_punishment += continuous
        self.cum_wall_punishment += wall
        self.cum_exploration_reward += exploration
        self.cum_movement_reward += movement
        self.cum_revisit_penalty += revisit
        self.cum_total_reward += actual_total

        # Update recent values for calculating averages
        self.recent_values['continuous'].append(continuous)
        self.recent_values['wall'].append(wall)
        self.recent_values['exploration'].append(exploration)
        self.recent_values['movement'].append(movement)
        self.recent_values['revisit'].append(revisit)
        self.recent_values['total'].append(actual_total)

        # Update min/max tracking
        self._update_min_max('continuous', continuous)
        self._update_min_max('wall', wall)
        self._update_min_max('exploration', exploration)
        self._update_min_max('movement', movement)
        self._update_min_max('revisit', revisit)
        self._update_min_max('total', actual_total)

        # Increment counter and print summary if interval reached
        self.step_counter += 1
        if self.step_counter % self.print_interval == 0:
            self._print_summary()

    def _update_min_max(self, component, value):
        """Update min/max values for a component"""
        if value < self.min_values[component]:
            self.min_values[component] = value
        if value > self.max_values[component]:
            self.max_values[component] = value

    def _print_summary(self):
        """Print a minimal summary of reward components with corrected calculations"""
        # Calculate averages of recent values
        avg_continuous = self._calculate_average('continuous')
        avg_wall = self._calculate_average('wall')
        avg_exploration = self._calculate_average('exploration')
        avg_movement = self._calculate_average('movement')
        avg_revisit = self._calculate_average('revisit')
        avg_total = self._calculate_average('total')

        # Print minimal summary
        print("\nREWARD SUMMARY (Episode {})".format(self.episode_number))
        print(f"Continuous:  avg {avg_continuous:.2f} | cum {self.cum_continuous_punishment:.2f}")
        print(f"Wall:        avg {avg_wall:.2f} | cum {self.cum_wall_punishment:.2f}")
        print(f"Exploration: avg {avg_exploration:.2f} | cum {self.cum_exploration_reward:.2f}")
        print(f"Movement:    avg {avg_movement:.2f} | cum {self.cum_movement_reward:.2f}")
        print(f"Revisit:     avg {avg_revisit:.2f} | cum {self.cum_revisit_penalty:.2f}")
        print(f"TOTAL:       avg {avg_total:.2f} | cum {self.cum_total_reward:.2f}")
        print(f"Step count:  {self.step_counter}")

    def _calculate_average(self, key):
        """Calculate average of recent values with error handling"""
        values = self.recent_values[key]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def reset_data(self):
        """Completely reset all tracking data for a new episode"""
        # Print final summary of the completed episode
        print("\n--- EPISODE {} COMPLETE - REWARDS RESET ---\n".format(self.episode_number))
        self._print_summary()

        # Increment episode number
        self.episode_number += 1

        # Reset step counter
        self.step_counter = 0

        # IMPORTANT: Fully clear ALL cumulative values
        self.cum_continuous_punishment = 0.0
        self.cum_wall_punishment = 0.0
        self.cum_exploration_reward = 0.0
        self.cum_movement_reward = 0.0
        self.cum_revisit_penalty = 0.0
        self.cum_total_reward = 0.0

        # Clear recent values
        for key in self.recent_values:
            self.recent_values[key].clear()

        # Reset min/max tracking
        self.min_values = {'continuous': 0, 'wall': 0, 'exploration': 0, 'movement': 0, 'revisit': 0, 'total': 0}
        self.max_values = {'continuous': 0, 'wall': 0, 'exploration': 0, 'movement': 0, 'revisit': 0, 'total': 0}

        print("\n--- STARTING EPISODE {} ---\n".format(self.episode_number))