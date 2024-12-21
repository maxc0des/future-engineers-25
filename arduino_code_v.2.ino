#include <Wire.h>

#define address 0x04

int value;
int target;
int response;
byte received_data[4];
int data_index = 0;
int motor_speed = 0;
int steering_angel = 0;

void receive_data() {
  while (Wire.available()) { //incoming data is saved into an array
    if (data_index < 4) {
      received_data[data_index] = Wire.read();
      data_index++;
    }
    if (data_index == 4) { //check whether all related 4 values ​​are in one array
      target = received_data[0]; // array is seperated into the target motor and the value that is supposed to be written
      value = received_data[1] * 100 + received_data[2] * 10 + received_data[3];
      response = check_data(received_data);
      process_input(received_data);
      data_index = 0;
    }
  }
}

//check if the data we recived makes sense
bool check_data(byte data[4]) {
  if ((target == 1 || target == 2 || target == 3) &&
      (value < 255)){
    return true;
  }
  return false;
}

void send_response() {
  Wire.write(response);
}

void process_input(byte data[4]) {
  //the follwoing lines are just for debugging
  Serial.println("Received data:");

  for (int i = 0; i < 4; i++) {
    Serial.print(data[i]);
    Serial.print(" ");
  }
  Serial.println();
  
  // Verarbeiten des Motors basierend auf der ersten Zahl
  if (data[0] == '1') {  // Servo
    steering_angel = value;
    Serial.print("Servo angle: ");
    Serial.println(steering_angel);
  }
  else if (data[0] == '2') {  // Schrittmotor
    motor_speed = value;
    Serial.print("Motor speed: ");
    Serial.println(motor_speed);
  }
}

/*void drive(int direction, int speed, int angel){
  if (dircetion == 0) {
  //code to stop motor
  }
  else if (direction == 1) {
  //code to drive backwards
  }
  else if (direction == 2) {
  //code to drive forward
  }
  digitalWrite(speed_pin, speed);
  //code to set servo to angel
}*/

void setup() {
  Serial.begin(115200);
  Wire.begin(address);
  Wire.onReceive(receive_data);
  Wire.onRequest(send_response);
}

void loop() {
  delay(10);
}