from get_data import *
counter = 0
root = os.getcwd()
while True:
    try:
        sensor = input("which sensor do you want to read?")
        if sensor == "tof":
            print(get_tof())
        elif sensor == "cam":
            filepath = take_photo(filepath=root, index=counter)
            print(f"photo saved at {filepath}")
            counter += 1
        else:
            print("invalid input")
    except KeyboardInterrupt:
        print("\nexiting")
        break