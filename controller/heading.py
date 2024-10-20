import math 
import numpy as np

def generate_straight_line(start, stop, spacing = 0.05):
    # Generate the points for the straight line
    num_lines = round(math.sqrt(abs(stop[0] - start[0]) ** 2 + abs(stop[1] - start[1]) ** 2) / spacing)

    # Generate the x and y points
    x_points = np.linspace(start[0], stop[0], num_lines)
    y_points = np.linspace(start[1], stop[1], num_lines)
    heading = math.atan2((stop[1] - start[1]), (stop[0] - start[0]))

    heading -= math.pi / 2 # adjust for the robot's orientation

    # Adjust the heading to be between 0 and 2pi
    if heading < 0:
        heading += 2 * math.pi

    # convert to degrees
    heading = math.degrees(heading)

    # Combine the points so that it is (x, y, heading)
    points = np.column_stack((x_points, y_points, np.ones(num_lines) * heading))

    return points
