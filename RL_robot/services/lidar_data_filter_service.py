import numpy as np


def lidar_scan_filter(ranges, range_max):
    """Process laser scan data"""
    # Divide the scan into 16 sectors and get min distance for each sector
    valid_ranges = np.where(np.isfinite(ranges), ranges, range_max)

    # Split the scan into 16 equal sectors
    num_sectors = 16
    sector_size = len(valid_ranges) // num_sectors

    measured_distance_to_walls = []
    for i in range(num_sectors):
        start_idx = i * sector_size
        end_idx = (i + 1) * sector_size if i < num_sectors - 1 else len(valid_ranges)
        sector_ranges = valid_ranges[start_idx:end_idx]
        min_distance = np.min(sector_ranges)
        measured_distance_to_walls.append(float(min_distance))

    return measured_distance_to_walls