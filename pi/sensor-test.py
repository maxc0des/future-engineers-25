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
        elif sensor == "fast cam":
            print(take_photo_fast())
        elif sensor == "test":
            if take_photo_fast() is not None and get_tof() is not None:
                print("test successful")
            else:
                print("test failed")
            break
        elif sensor == "gyro":
            angle = get_gyro()
            print(angle)
        else:
            print("invalid input")
    except KeyboardInterrupt:
        print("\nexiting")
        break