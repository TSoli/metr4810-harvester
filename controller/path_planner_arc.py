import numpy as np
import math
from utils.path_planning_visualiser import visualize_segments 

class PathPlannerArc:

    def __init__(self,start_x, start_y, scoop_width, overlap_percentage: int, travel_radius):

        self.paths = [] # list of numpy arrays. 
        self.start_x = start_x
        self.start_y = start_y
        self.scoop_width = scoop_width
        self.overlap_percentage = overlap_percentage
        self.radius = travel_radius

    def get_single_search_sweep_angle(self):
        return self.scoop_width / self.radius
    
    def get_single_search_sweep_area(self):
        return 0.5 * self.get_single_search_sweep_angle() * (self.radius ** 2)
    
    def get_number_of_segments(self):
        quadrant_area = (math.pi * self.radius ** 2) * 0.25
        return math.ceil((quadrant_area / self.get_single_search_sweep_area()) * (1 + self.overlap_percentage / 100))
    
    def get_adjusted_theta(self):
        segments = pathPlanner.get_number_of_segments()
        return math.radians(90) / segments
    
    def get_all_radii_angles(self):
        radii_angles = []
        adjusted_theta = self.get_adjusted_theta()
        for i in range(self.get_number_of_segments()):
            radii_angles.append((i * adjusted_theta) + adjusted_theta / 2)

        return radii_angles
    
    def generate_points(x_start, y_start, angle_rad, separation, end_radius):
        points = []
        radius = 0
        current_x = x_start
        current_y = y_start
        
        while radius <= end_radius:
            w = angle_rad + math.radians(270)  # Adjusting w by radians(270)
            points.append([current_x, current_y, w])
            
            # Move to the next point
            current_x += separation * math.cos(angle_rad)
            current_y += separation * math.sin(angle_rad)
            
            # Update the radius
            radius = math.sqrt((current_x - x_start)**2 + (current_y - y_start)**2)
        
        return np.array(points)


# Example usage
x_start = 0
y_start = 0
angle_rad = math.radians(45)  # 45 degrees in radians
separation = 1.0
end_radius = 5.0

#points_array = generate_points(x_start, y_start, angle_rad, separation, end_radius)
#print(points_array)
    
pathPlanner = PathPlannerArc(100, 100, 200, 0, 1800)

segments = pathPlanner.get_number_of_segments()

angles = pathPlanner.get_all_radii_angles()
for angle in angles:
    print(math.degrees(angle))




    
    
   

