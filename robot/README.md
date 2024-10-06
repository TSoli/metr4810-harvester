# Pico W Multicore Operation + Networking

Architecture:

- Core 0: Server
- Core 1: GPIO processor.

A shared data strcuture like a queue can be used to send data between threads.

## Setting up pico W as Wifi Access Point 

- The `network` module in micropython allows the pico to act as a wifi access point to allow devices like a laptop to connect directly to the picos network so it can rceieve and respond to requests.
- The default IPv4 address is `192.168.4.1`. This may be soemthing want to change to avoid getting requests from other people systems during operation. 

Use the '_thread' library to run a program on the pico's second core.

Simple Example: https://www.youtube.com/watch?v=mm1EoNqjd4c

## Example Code

This code is AI generated but serves as an example how we could receive POST requests containing JSON and process them on the second core:

```

import network
import socket
import _thread
import machine
import ujson
from time import sleep
from collections import deque

# Shared queue for communication between Core 0 and Core 1
command_queue = deque()

# GPIO setup (example)
led = machine.Pin(25, machine.Pin.OUT)

# Wi-Fi Access Point setup (Core 0)
def create_access_point():
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    ap.config(essid="PicoW-POSTAP", password="12345678")
    ap.ifconfig(('192.168.10.1', '255.255.255.0', '192.168.10.1', '8.8.8.8'))
    print("Access Point created with IP:", ap.ifconfig())

# HTTP server to listen for POST requests (Core 0)
def server_thread():
    addr = socket.getaddrinfo('0.0.0.0', 80)[0][-1]
    s = socket.socket()
    s.bind(addr)
    s.listen(1)
    print('Listening on', addr)

    while True:
        cl, addr = s.accept()
        print('Client connected from', addr)
        request = cl.recv(1024).decode('utf-8')

        # Check if it's a POST request
        if "POST" in request:
            # Parse request body (assume it's JSON for this example)
            content_length = int(request.split("Content-Length: ")[1].split("\r\n")[0])
            body = cl.recv(content_length).decode('utf-8')
            
            # Decode JSON data from POST request
            try:
                data = ujson.loads(body)
                print('Received data:', data)

                # Add the command to the shared queue for Core 1 to process
                if "command" in data:
                    command_queue.append(data['command'])

                # Send response to client
                response = 'HTTP/1.1 200 OK\n\nCommand received'
            except Exception as e:
                print("Error decoding request:", e)
                response = 'HTTP/1.1 400 Bad Request\n\nError decoding JSON'
        
        else:
            response = 'HTTP/1.1 405 Method Not Allowed\n\nOnly POST allowed'

        cl.send(response)
        cl.close()

# GPIO handler running on Core 1
def gpio_thread():
    while True:
        if command_queue:
            command = command_queue.popleft()
            if command == 'on':
                led.on()
                print('LED ON (Processed by Core 1)')
            elif command == 'off':
                led.off()
                print('LED OFF (Processed by Core 1)')
            else:
                print('Unknown command:', command)
        sleep(0.1)  # Avoid busy waiting

# Start the access point and server
create_access_point()
_thread.start_new_thread(server_thread, ())

# Start GPIO processing on Core 1
_thread.start_new_thread(gpio_thread, ())

while True:
    # Main thread can do other tasks, if needed
    sleep(1)


```

Then a post request could be made to `http://192.168.10.1` with this data:
```
{
  "command": "on"
}
```







