CROP_SIZE_METERS = 6.0  # 6m x 6m area


def calc_map_center(origin_x, origin_y, width, height, resolution, odom_ready, pos, slam_pose):
    center_cell_x = 0
    center_cell_y = 0
    if slam_pose is not None:
        # Convert SLAM position to grid cell coordinates
        center_cell_x = int((slam_pose[1] - origin_x) / resolution)
        center_cell_y = int((slam_pose[2] - origin_y) / resolution)
        print(f"Updated map center using SLAM pose: ({center_cell_x}, {center_cell_y})")
    elif odom_ready:
        # Fall back to odometry if SLAM not available
        enter_cell_x = int((pos[1] - origin_x) / resolution)
        center_cell_y = int((pos[2] - origin_y) / resolution)
        print(f"Updated map center using odometry: ({center_cell_x}, {center_cell_y})")
    else:
        # If no position data, use the center of the map
        center_cell_x = width // 2
        center_cell_y = height // 2
        print(f"Updated map center using map center: ({center_cell_x}, {center_cell_y})")

    return center_cell_x, center_cell_y


def calc_map_range(center_cell_x, center_cell_y, height, resolution, width):
    # Calculate size of 6m x 6m area in grid cells (maintaining resolution)
    crop_size_cells = int(CROP_SIZE_METERS / resolution)

    # Ensure crop_size_cells is even for better centering
    if crop_size_cells % 2 != 0:
        crop_size_cells += 1

    # Calculate boundaries for cropping
    half_size = crop_size_cells // 2
    min_x = max(0, center_cell_x - half_size)
    min_y = max(0, center_cell_y - half_size)

    # Instead of truncating at the edge, we shift the window to fully fit within bounds
    if min_x + crop_size_cells > width:
        min_x = max(0, width - crop_size_cells)
    if min_y + crop_size_cells > height:
        min_y = max(0, height - crop_size_cells)

    # Calculate max coordinates based on the fixed crop size
    max_x = min_x + crop_size_cells
    max_y = min_y + crop_size_cells

    # Debug output
    # print(f"Cropping map: [{min_x}:{max_x}, {min_y}:{max_y}] from original {width}x{height}")
    # print(f"Crop dimensions: {max_x - min_x}x{max_y - min_y}")

    return crop_size_cells, max_x, max_y, min_x, min_y


def crop_map(data_map, width, height, resolution, center_cell_x, center_cell_y):
    """Process SLAM map data by cropping a 6m x 6m area centered on the robot's starting position"""

    crop_size_cells, max_x, max_y, min_x, min_y = calc_map_range(center_cell_x, center_cell_y, height, resolution, width)

    # Create empty cropped map with the correct size
    cropped_map = []

    # Extract the map data cells, ensuring we stay in bounds
    for y in range(min_y, max_y):
        row_data = []  # Store row for debugging
        for x in range(min_x, max_x):
            if 0 <= y < height and 0 <= x < width:
                idx = y * width + x
                if idx < len(data_map):
                    if data_map[idx] == -1:  # Unknown
                        cell_value = -1.0
                    else:  # 0-100 scale to 0-1
                        cell_value = float(data_map[idx]) / 100.0
                else:
                    # This should not happen if our bounds checking is correct
                    cell_value = -1.0
                    print(f"Warning: Index {idx} out of bounds for data_map (len={len(data_map)}) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! shouldnt happen prob bug in crop map, has never happened before")
            else:
                # Out of bounds of the original map
                cell_value = -1.0
                # print(f"Warning: Coordinates ({x},{y}) out of bounds for original map {width}x{height}")

            cropped_map.append(cell_value)
            row_data.append(cell_value)

        # Debug: print first and last row
        # if y == min_y or y == max_y - 1:
        #     print(f"Row {y - min_y} data sample: {row_data[:5]}...")

    # Verify the size of the cropped map
    expected_size = crop_size_cells * crop_size_cells
    actual_size = len(cropped_map)
    if actual_size != expected_size:
        # print(f"WARNING: Unexpected cropped map size. Expected {expected_size}, got {actual_size}")
        # Ensure correct size by padding/truncating if needed
        if actual_size < expected_size:
            cropped_map.extend([-1.0] * (expected_size - actual_size))
        else:
            cropped_map = cropped_map[:expected_size]

    return cropped_map