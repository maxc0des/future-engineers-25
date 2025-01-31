from pi.motor import *

setup()

while True:
    try:
        mode = input("choose mode (motor/servo)")
        if mode == motor:
            speed = int(input("choose speed (-255 - 255)"))
            motor(speed)
        elif mode == servo:
            angle = int(input("choose angle (0 - 180)"))
            servo(angle)
        else:
            print("invalid input")
    except KeyboardInterrupt:
        cleanup()
        print("stopped the test")
        break