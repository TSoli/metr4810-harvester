from pynput import keyboard
import time

high_ground_request = False
return_to_delivery_point_request = False
dispense_beans_request = False
start_deployment_request = False
stop_request_made = False # Safety

HIGH_GROUND_KEY = "h"
DELIVERY_REQUEST_KEY = "d"
DISPENSE_BEANS = "b"
START_KEY = "s"



def on_press(key):
    global stop_request_made, high_ground_request, start_deployment_request
    global return_to_delivery_point_request, dispense_beans_request

    try:
        if key.char == HIGH_GROUND_KEY:
            high_ground_request = True
        elif key.char == START_KEY:
            start_deployment_request = True
        elif key.char == DELIVERY_REQUEST_KEY:
            return_to_delivery_point_request = True
        elif key.char == DISPENSE_BEANS:
            dispense_beans_request = True
        else:
            print("unknown key")

    except Exception as e:
        print("error:", e)
        #print(f'Special key {key} pressed')

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False
           

def main():
    # Collect events until released
    listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
    listener.start()
    global stop_request_made, high_ground_request, start_deployment_request
    global return_to_delivery_point_request, dispense_beans_request
    while True:
        if stop_request_made:
            print("stop vehicle")
            stop_request_made = False
        
        if high_ground_request:
            print("go to high ground")
            high_ground_request = False
        
        if start_deployment_request:
            print("start deployment")
            start_deployment_request = False
        
        if return_to_delivery_point_request:
            print("Return to delivery point")
            return_to_delivery_point_request = False

        if dispense_beans_request:
            print("Dispense beans")
            dispense_beans_request = False
        time.sleep(1)h

if __name__ == "__main__":
    main()