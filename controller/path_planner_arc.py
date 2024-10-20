import numpy as np
import math
from utils.path_planning_visualiser import visualize_segments

class PathPlannerArc:

    def __init__(self,start_x, start_y, scoop_width, overlap_percentage: int, travel_radius, waypoint_seperation=None):

        self.paths = [] # list of numpy arrays. 
        self.start_x = start_x
        self.start_y = start_y
        self.scoop_width = scoop_width
        self.overlap_percentage = overlap_percentage
        self.radius = travel_radius
        self.waypoint_seperation = waypoint_seperation if waypoint_seperation else travel_radius

    def get_single_search_sweep_angle(self):
        return self.scoop_width / self.radius
    
    def get_single_search_sweep_area(self):
        return 0.5 * self.get_single_search_sweep_angle() * (self.radius ** 2)
    
    def get_number_of_segments(self):
        quadrant_area = (math.pi * self.radius ** 2) * 0.25
        return math.ceil((quadrant_area / self.get_single_search_sweep_area()) * (1 + self.overlap_percentage / 100))
    
    def get_adjusted_theta(self):
        segments = self.get_number_of_segments()
        return math.radians(90) / segments
    
    def get_all_radii_angles(self):
        radii_angles = []
        adjusted_theta = self.get_adjusted_theta()
        for i in range(self.get_number_of_segments()):
            radii_angles.append((i * adjusted_theta) + adjusted_theta / 2)

        return radii_angles
    
    def generate_outward_radial_points(self, angle_rad):
        points = []
        radius = 0
        current_x = self.start_x
        current_y = self.start_y
        
        while radius <= self.radius:
            w = angle_rad + math.radians(270)  # Adjusting w by radians(270)
            points.append([current_x, current_y, w])
            
            # Move to the next point
            current_x += self.waypoint_seperation * math.cos(angle_rad)
            current_y += self.waypoint_seperation * math.sin(angle_rad)
            
            # Update the radius
            radius = math.sqrt((current_x - self.start_x)**2 + (current_y - self.start_y)**2)
        
        return np.array(points)
    

    def generate_inwards_radial_points(self, angle_rad):
        points = []
        radius = self.radius
        current_x = self.start_x + radius * math.cos(angle_rad)
        current_y = self.start_y + radius * math.sin(angle_rad)

        while radius >= 0:
            w = angle_rad + math.radians(270)  # Adjusting w by radians(270)
            points.append([current_x, current_y, w])

            # Move to the next point inward
            current_x -= self.waypoint_seperation * math.cos(angle_rad)
            current_y -= self.waypoint_seperation * math.sin(angle_rad)
            
            if (current_x < self.start_x) or (current_y < self.start_y):
                break

            # Update the radius
            radius = math.sqrt((current_x - self.start_x)**2 + (current_y - self.start_y)**2)

        points.append([self.start_x, self.start_y, angle_rad + math.radians(270)])
            

        return np.array(points)

    def generate_all_radial_segments(self):
        angles = self.get_all_radii_angles()
        segments = []
        for angle in angles:
            segments.append(self.generate_outward_radial_points(angle))
            segments.append(self.generate_inwards_radial_points(angle))
        return segments

# Plot the segment
if __name__ == "__main__":
    pathPlanner = PathPlannerArc(0.3, 0.3, 0.18, 10, 1.4, 0.05)
    segments = pathPlanner.generate_all_radial_segments()
    visualize_segments(segments)