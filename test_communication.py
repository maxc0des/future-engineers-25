import random
from send_i2c import send_i2c

def generate_values():
    values = []
    for i in range(4):
        values.append(random.randint(1,2))
    print(values)
    return values

while 1:
    try:
        input("press enter to send next test data")
        send_i2c(generate_values())
    except KeyboardInterrupt:
        print("ending the test")
        break