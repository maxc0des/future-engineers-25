import time
from picamera2 import Picamera2

picam2 = Picamera2()
counter = 0

config = picam2.create_still_configuration()
picam2.configure(config)

picam2.start()
time.sleep(2)

while True:
    try:
        input("")
        counter += 1
        picam2.capture_file(f"test-picture{counter}.jpg")
    
    except KeyboardInterrupt:
        break