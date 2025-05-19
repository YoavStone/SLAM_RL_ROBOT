def calc_grid_pos(position, grid_x, grid_y, center_cell_x, center_cell_y, width, height, crop_size_cells):
    """
    Places the robot at a consistent position in the cropped map based on
    its offset from the center position
    """
    _, x, y = position

    # Calculate crop boundaries
    half_size = crop_size_cells // 2
    min_x = max(0, center_cell_x - half_size)
    min_y = max(0, center_cell_y - half_size)

    # Adjust for map boundaries (same logic as in map_callback)
    if min_x + crop_size_cells > width:
        min_x = max(0, width - crop_size_cells)
    if min_y + crop_size_cells > height:
        min_y = max(0, height - crop_size_cells)

    # Adjust to coordinates within the cropped map
    grid_x = grid_x - min_x
    grid_y = grid_y - min_y

    # Ensure coordinates are within bounds of the cropped map
    grid_x = max(0, min(crop_size_cells - 1, grid_x))
    grid_y = max(0, min(crop_size_cells - 1, grid_y))

    # Return with grid position
    return float(grid_x), float(grid_y)