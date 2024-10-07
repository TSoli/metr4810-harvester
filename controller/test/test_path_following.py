import os
import sys

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_directory)

import numpy as np
from path_following import PurePursuitController


def main():
    ppc = PurePursuitController(5, 1)
    path = np.array([[1, 1, -np.pi / 4]])
    ppc.path = path
    # expect 1, 0
    print(ppc.get_control_action(np.array([0, 0, -np.pi / 4])))
    # expect 1, -1
    print(ppc.get_control_action(np.array([0, 0, 0])))
    # expect -1 ,0
    print(ppc.get_control_action(np.array([0, 0, 3 * np.pi / 4])))
    # expect 1, 0.4
    path = np.array([[-1, 2, 0]])
    ppc.path = path
    print(ppc.get_control_action(np.array([0, 0, 0])))
    # expect 1, 1
    print(ppc.get_control_action(np.array([0, 1, 0])))


if __name__ == "__main__":
    main()
