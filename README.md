# metr4810-harvester

Code for METR4810 2024S2 Sand Harvester project

## Getting Started

The software is written using Python 3.11 and
[MicroPyton](https://micropython.org/) on a Raspberry Pi Pico.

### Programming the Robot

1. First install the
   [MicroPython firmware for the Raspberry Pi Pico](https://micropython.org/download/RPI_PICO/).

2. Create a virtual environment for Python and install the required packages.

```sh
pyton -m venv robot/venv
source robot/venv/bin/activate # or similar for Windows
pip install -r robot/requirements.txt
```

3. Upload the software files to the robot.

```sh
rshell
cp -r robot/* /pyboard/
```

Note the file named `main.py` is the script that will run automatically after
boot.

### Setting UP NRF24L01 + PICO

Tutorial:
https://coffeebreakpoint.com/micropython/how-to-connect-a-nrf24l01-transceiver-to-your-raspberry-pi-pico/
