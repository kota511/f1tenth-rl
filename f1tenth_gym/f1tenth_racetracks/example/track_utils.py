import numpy as np

def compute_track_length(waypoints, x_index, y_index, closed_loop=True):
    """
    Computes the total track length given the waypoints.
    
    Parameters:
    - waypoints: A numpy array containing the waypoint information.
    - x_index: The index of the x-coordinate in the waypoints.
    - y_index: The index of the y-coordinate in the waypoints.
    - closed_loop: If True, adds the distance from the last waypoint back to the first.
    
    Returns:
    - total_length: The total track length (float).
    """
    # Extract the x and y columns from waypoints
    points = waypoints[:, [x_index, y_index]]
    
    # Compute the distance between each consecutive pair of points.
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = segment_lengths.sum()
    
    if closed_loop:
        # Add the distance between the last and first waypoint.
        total_length += np.linalg.norm(points[-1] - points[0])
    
    return total_length
