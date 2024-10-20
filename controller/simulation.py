import matplotlib.pyplot as plt
import matplotlib.animation as animation

class RobotController:
    def __init__(self, initial_pose):
        self._pose = initial_pose
        self._stored_pose = initial_pose
        self._has_been_moved = False

    def get_stored_pose(self):
        return self._stored_pose

    def set_current_pose(self, pose):
        self._pose = pose

    def get_current_pose(self):
        return self._pose

    def set_has_been_moved(self):
        self._has_been_moved = True

    def clear_has_been_moved(self):
        self._has_been_moved = False

    def update_pose(self, control_action):
        # Assuming control_action is a tuple (dx, dy)
        x, y = self._pose
        dx, dy = control_action
        new_pose = (x + dx, y + dy)
        self.set_current_pose(new_pose)
        self.set_has_been_moved()
        self._pose = new_pose

# Define the initial pose of the robot
initial_pose = (0, 0)
robot = RobotController(initial_pose)

# Define a list of control actions (dx, dy)
control_actions = [(-0.1, 0.1), (2, 1), (-1, 0), (0, -1), (1, 1), (-1, -1)]

# Initialize the plot
fig, ax = plt.subplots()
x_coords, y_coords = [initial_pose[0]], [initial_pose[1]]
line, = ax.plot(x_coords, y_coords, marker='o')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Robot Path Simulation')
ax.grid(True)

# Update function for animation
def update(frame):
    action = control_actions[frame % len(control_actions)]
    robot.update_pose(action)
    x, y = robot.get_current_pose()
    x_coords.append(x)
    y_coords.append(y)
    line.set_data(x_coords, y_coords)
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(control_actions), interval=500, blit=True, repeat=True)

plt.show()