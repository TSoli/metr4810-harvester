import math

import numpy as np
from utils.path_planning_visualiser import (visualize_segments,
                                            visualize_segments_zig_zag)


class PathPlannerZigZag:
    def __init__(
        self,
        coverage_width,
        coverage_height,
        waypoint_spacing,
        start_x,
        start_y,
        scoop_width,
        overlap_percentage: int,
    ):
        self.coverage_width = coverage_width
        self.coverage_height = coverage_height
        self.paths = []  # list of numpy arrays.
        self.spacing = waypoint_spacing
        self.start_x = start_x
        self.start_y = start_y
        self.scoop_width = scoop_width
        self.overlap_percentage = overlap_percentage

    def get_number_of_segments(self):
        return len(self.paths)

    def get_segmented_paths(self):
        return self.paths

    def generate_straight_segment(self, start, stop):
        points = np.linspace(start, stop, round(abs(stop - start) / self.spacing))
        return points

    def generate_horizontal_segment(
        self, y_coordinate, start_x, stop_x, moving_in_positive_axis_direction: bool
    ):
        omega = 0
        if moving_in_positive_axis_direction:
            omega = math.radians(-90)
        else:
            omega = math.radians(90)

        straight_x_segments = self.generate_straight_segment(start_x, stop_x)
        return np.array([(xi, y_coordinate, omega) for xi in straight_x_segments])

    def generate_vertical_segment(
        self, x_coordinate, start_y, stop_y, moving_in_positive_axis_direction: bool
    ):
        omega = 0
        if moving_in_positive_axis_direction:
            omega = math.radians(0)
        else:
            omega = math.radians(180)

        straight_y_segments = self.generate_straight_segment(start_y, stop_y)
        return np.array([(x_coordinate, yi, omega) for yi in straight_y_segments])

    def get_number_of_horizontal_passes(self):
        return round(
            ((self.overlap_percentage / 100) + 1)
            * (self.coverage_height / self.scoop_width)
        )

    def get_y_coordinates_for_horizontal_passes(self):
        num_required_passes = self.get_number_of_horizontal_passes()

        y_coordinates = []
        for i in range(num_required_passes):
            y_coordinates.append(
                (i * self.scoop_width + self.scoop_width / 2) + self.start_y
            )

        return y_coordinates

    def get_number_of_vertical_passes(self):
        return self.get_number_of_horizontal_passes() - 1

    def get_x_coordinates_for_vertcal_passes(self):
        num_vertical_passes = self.get_number_of_vertical_passes()
        right_vertical_segment = True
        x_coordinates = [self.start_x]
        for i in range(num_vertical_passes):
            x_coordinates.append(
                float(self.coverage_width) + self.start_x
                if right_vertical_segment
                else self.start_x
            )
            right_vertical_segment = not right_vertical_segment
        return x_coordinates

    def generate_zigzag_path(self):
        y_coordinates = self.get_y_coordinates_for_horizontal_passes()
        start_x, end_x = self.start_x, self.start_x + self.coverage_width
        y_forward = True
        x_forward = True
        for y_idx in range(1, len(y_coordinates)):
            self.paths.append(
                self.generate_horizontal_segment(
                    y_coordinates[y_idx - 1], start_x, end_x, x_forward
                )
            )
            x_forward = not x_forward
            self.paths.append(
                self.generate_vertical_segment(
                    end_x, y_coordinates[y_idx - 1], y_coordinates[y_idx], y_forward
                )
            )
            start_x, end_x = end_x, start_x
            self.paths.append(
                self.generate_horizontal_segment(
                    y_coordinates[y_idx], start_x, end_x, x_forward
                )
            )

        return self.paths
