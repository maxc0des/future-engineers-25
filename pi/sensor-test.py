from get_data import *

while True:
    try:
        sensor = input("which sensor do you want to read?")
        if sensor == "tof":
            print(get_tof())
        elif sensor == "cam":
            filepath = take_photo()
            print(f"photo saved at {filepath}")
        else:
            print("invalid input")
    except KeyboardInterrupt:
        print("exiting\n")
        break