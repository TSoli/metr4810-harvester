import matplotlib.pyplot as plt
import random

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
        plt.plot(x_values, y_values, color=colors(idx))
        
        # Label the segment in the middle
        mid_index = len(segment) // 2
        plt.text(x_values[mid_index], y_values[mid_index], f'Segment {idx+1}', fontsize=9, color=colors(idx))
        
        # Only label the start point for the first array
        if idx == 0:
            plt.scatter(x_values[0], y_values[0], color='green', zorder=5)
            plt.text(x_values[0], y_values[0], 'Start point', fontsize=9, ha='right', color='green')
        
        # Only label the stop point for the last array
        if idx == len(segment_list) - 1:
            plt.scatter(x_values[-1], y_values[-1], color='red', zorder=5)
            plt.text(x_values[-1], y_values[-1], 'Stop point', fontsize=9, ha='right', color='red')
    
    # Add labels
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title('Visualization of Segments with Single Start/Stop Points')
    
    # Show the plot
    plt.show()