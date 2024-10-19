import numpy as np
import math
import matplotlib.pyplot as plt
import random


class PathPlanner:

    def __init__(self, width, height, waypoint_spacing):
        self.width = width
        self.height = height
        self.last_pose = ()
        self.current_waypont = None
        self.paths = [] # list of numpy arrays. 
        self.num_segments = 0
        self.spacing = waypoint_spacing

    def get_last_x(self):
        return self.last_pose[0]
    
    def get_last_y(self):
        return self.last_pose[1]

    def generate_straight_segment(self, start, stop):
        points = np.linspace(start, stop, round(abs(stop - start) / self.spacing))
        #print(points)
        return points
    
    def generate_horizontal_segment(self, y_coordinate, start_x, stop_x, moving_in_positive_axis_direction: bool):
        omega = 0
        if (moving_in_positive_axis_direction):
            omega = math.radians(270)
        else:
            omega = math.radians(90)
        
        straight_x_segments = self.generate_straight_segment(start_x, stop_x)
        return np.array([(xi, y_coordinate, omega) for xi in straight_x_segments])
    
    def generate_vertical_segment(self, x_coordinate, start_y, stop_y, moving_in_positive_axis_direction: bool):
        omega = 0
        if (moving_in_positive_axis_direction):
            omega = math.radians(0)
        else:
            omega = math.radians(180)
        
        straight_y_segments = self.generate_straight_segment(start_y, stop_y)
        return np.array([(x_coordinate, yi, omega) for yi in straight_y_segments])
    
    def get_number_of_horizontal_passes(self, scoop_width, coverage_height, overlap_percentage: int):
        return round(((overlap_percentage / 100) + 1) * (coverage_height / scoop_width))
    
    def get_y_coordinates_for_horizontal_passes(self, scoop_width, start_y, end_y, overlap_percentage: int):

        num_required_passes = self.get_number_of_horizontal_passes(scoop_width, abs(end_y - start_y), overlap_percentage)

        y_coordinates = []
        for i in range(num_required_passes):
            y_coordinates.append(i * scoop_width + scoop_width / 2)

        return y_coordinates
    
        
    def get_number_of_vertical_passes(self, scoop_width, coverage_height, overlap_percentage: int):
        return self.get_number_of_horizontal_passes(scoop_width, coverage_height, overlap_percentage) - 1
    
    def get_x_coordinates_for_vertcal_passes(self, start_x, end_x, coverage_height, scoop_width, overlap_percentage):
        coverage_width = end_x - start_x
        num_vertical_passes = self.get_number_of_vertical_passes(scoop_width, coverage_height, overlap_percentage)
        right_vertical_segment = True
        x_coordinates = [start_x]
        for i in range(num_vertical_passes):
            x_coordinates.append(float(coverage_width) + start_x if right_vertical_segment else start_x)
            right_vertical_segment = not right_vertical_segment
        return x_coordinates
    

    
    #def generate_overall_path()
    


planner = PathPlanner(2000, 2000, 50)

#planner.generate_straight_segment(50, 1950)

#planner.generate_straight_segment(1950, 50)
#actually 180
y_coordinates = planner.get_y_coordinates_for_horizontal_passes(200, 50, 1950, 0)
x_coordinates = planner.get_x_coordinates_for_vertcal_passes(50, 1950, 1950, 200, 0)

print(f"X coordinates {x_coordinates}")
print(f"Y coordinates {y_coordinates}")
segmented_paths = []


planner.generate_horizontal_segment(y_coordinates[0], x_coordinates[0], x_coordinates[1], True)
planner.generate_vertical_segment(x_coordinates[1], y_coordinates[0], y_coordinates[1], True)
planner.generate_horizontal_segment(y_coordinates[1], x_coordinates[1], x_coordinates[0], False)

start_x, end_x = 50, 1950
y_forward = True
x_forward = True
print("hello")
for y_idx in range(1, len(y_coordinates)):
    print(f"Horizontal 1: x-start:{start_x}, x-end:{end_x}, y-coord={y_coordinates[y_idx - 1]}")
    segmented_paths.append(planner.generate_horizontal_segment(y_coordinates[y_idx - 1], start_x, end_x, x_forward))
    x_forward = not x_forward
    print(f"Vertical: x-point{end_x}, ystart={y_coordinates[y_idx - 1]} yend={y_coordinates[y_idx]}")
    segmented_paths.append(planner.generate_vertical_segment(end_x, y_coordinates[y_idx - 1], y_coordinates[y_idx], y_forward))
    #segmented_paths.append(planner.generate_vertical_segment(end_x, 100, 300, y_forward))
    start_x, end_x =  end_x, start_x
    print(f"Horizontal 2: x-start:{start_x}, x-end:{end_x}, y-coord={y_coordinates[y_idx]}")
    segmented_paths.append(planner.generate_horizontal_segment(y_coordinates[y_idx], start_x, end_x, x_forward))
    #start_x, end_x =  end_x, start_x
    #x_forward = not x_forward

def visualize_segments(segment_list):
    # Create a plot
    plt.figure(figsize=(8, 6))
    
    # Assign a unique color for each segment
    colors = plt.cm.get_cmap('hsv', len(segment_list))  # Using a colormap for unique colors
    
    # Loop through each segment (numpy array)
    for idx, segment in enumerate(segment_list):
        # Extract x and y coordinates, ignoring the last value
        x_values = segment[:, 0]
        y_values = segment[:, 1]
        
        # Plot the segment with a unique color
        plt.plot(x_values, y_values, color=colors(idx), label=f'Segment {idx+1}')
    
    # Add labels and a legend
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Visualization of Segments')
    plt.legend()
    
    # Show the plot
    plt.show()


visualize_segments(segmented_paths)

    


