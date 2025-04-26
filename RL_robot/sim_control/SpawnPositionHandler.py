import math
import random


class SpawnPositionHandler:
    def __init__(self, staring_pos):
        self.positions = [
            [0.0, 0.0],
            [6.3, 0.0],
            [-6.3, 0.0],
            [0.0, 6.3],
            [0.0, -6.3]
        ]
        try:
            x, y = staring_pos.split(',')
            self.staring_pos = [float(x.strip()), float(y.strip())]
        except ValueError:
            self.staring_pos = None
            print("no start pos specified teleport to random locations")
        self.target_spawn_position = None

    def get_target_position(self):
        """Get target position and yaw for teleportation"""
        # Random yaw angle
        yaw = random.choice([0.0, math.pi / 2, math.pi, math.pi * 3 / 2])

        if self.staring_pos is not None:
            # Use the provided spawn location if available
            return [self.staring_pos[0], self.staring_pos[1], yaw]
        else:
            # Choose a random position from predefined positions
            chosen_position = random.choice(self.positions)
            return [chosen_position[0], chosen_position[1], yaw]
