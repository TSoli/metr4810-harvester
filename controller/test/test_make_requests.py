import os
import sys

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_directory)

from robot_requests import *

def main():
    print("Test request functionality here")
    #test the comms here

if __name__ == "__main__":
    main()