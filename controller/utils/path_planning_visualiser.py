import matplotlib.pyplot as plt
import numpy as np
import math


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
        w_values = segment[:, 2] #These values are either 0 radians, pi/2 radians, pi radians, 3pi/2 radians, 
        
        # Plot the segment with a unique color
        plt.plot(x_values, y_values, color=colors(idx))
        
        # Plot each individual point as a small circle (marker='o')
        plt.scatter(x_values, y_values, color=colors(idx), s=20, label=f'Segment {idx+1}', zorder=3)
        
        # Label the segment in the middle
        mid_index = len(segment) // 2
        plt.text(x_values[mid_index], y_values[mid_index], f'Segment {idx+1}', fontsize=9, color=colors(idx))
        
        # Only label the start point for the first array
        if idx == 0:
            plt.scatter(x_values[0], y_values[0], color='green', s=50, zorder=5)
            plt.text(x_values[0], y_values[0], 'Start point', fontsize=9, ha='right', color='green')
        
        # Only label the stop point for the last array
        if idx == len(segment_list) - 1:
            plt.scatter(x_values[-1], y_values[-1], color='red', s=50, zorder=5)
            plt.text(x_values[-1], y_values[-1], 'Stop point', fontsize=9, ha='right', color='red')
    
    # Add labels
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Path Planning with Individual Points')
    
    # Show the plot
    plt.show()


def visualize_segments_zig_zag(segment_list):
    # Create a plot
    plt.figure(figsize=(8, 6))

    # Define colors for each angle
    angle_colors = {
        0: 'blue',           # For angle 0 radians
        math.pi / 2: 'green', # For angle π/2 radians
        math.pi: 'yellow',    # For angle π radians
        3 * math.pi / 2: 'purple'  # For angle 3π/2 radians
    }
    
    # Loop through each segment (numpy array)
    for idx, segment in enumerate(segment_list):
        # Extract x, y, and w (angle) coordinates
        x_values = segment[:, 0]
        y_values = segment[:, 1]
        w_values = segment[:, 2]  # These values are either 0, π/2, π, or 3π/2 radians
        
        # Get the unique color based on the angle (w) of the first point in the segment
        angle_w = w_values[0]  # Assuming all points in a segment have the same angle
        color = angle_colors.get(angle_w, 'black')  # Default to black if angle isn't found

        # Plot the segment with the corresponding color
        plt.plot(x_values, y_values, color=color)
        
        # Plot each individual point as a small circle (marker='o')
        plt.scatter(x_values, y_values, color=color, s=20, label=f'Segment {idx+1}', zorder=3)
        
        # Label the segment in the middle
        mid_index = len(segment) // 2
        plt.text(x_values[mid_index], y_values[mid_index], f'Segment {idx+1}', fontsize=9, color=color)
        
        # Only label the start point for the first array
        if idx == 0:
            plt.scatter(x_values[0], y_values[0], color='green', s=50, zorder=5)
            plt.text(x_values[0], y_values[0], 'Start point', fontsize=9, ha='right', color='green')
        
        # Only label the stop point for the last array
        if idx == len(segment_list) - 1:
            plt.scatter(x_values[-1], y_values[-1], color='red', s=50, zorder=5)
            plt.text(x_values[-1], y_values[-1], 'Stop point', fontsize=9, ha='right', color='red')
    
    # Add labels
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Path Planning with Angle-Based Colors')
    
    # Show the plot
    plt.show()