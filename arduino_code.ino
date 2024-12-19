// i looked over it and the code is not working yet with the pi, we will have to restructure it soon
#include <Wire.h>

#define address 9

int response;
char received_data[4];
int data_index = 0;
int motor_speed = 0;
int angel = 0;

void receive_data(){
  while (Wire.available()) {
    if (data_index < 4) { // jeweils die zusammengehörenden 4 zahlen in einem array speichern
      received_data[data_index] = Wire.read();
      data_index++;
    }

    if (data_index == 4) {
      if (check_data(received_data)) {
        response = 200;
        process_input(received_data);
      } else {
        response = 404;
      }
    }
  }
}

bool check_data(char data[4]) {
  // Überprüft, ob der erste Wert 1 oder 2 ist, um den richtigen Motor auszuwählen
  if (data[0] == '1' || data[0] == '2') {
    return true;
  } else {
    return false;
  }
}

void send_response() {
  Wire.write(response);
}

void process_input(char data[4]) {
  Serial.println("Received data:");

  // Debugging-Ausgabe der empfangenen Daten
  for (int i = 0; i < 4; i++) {
    Serial.print(data[i]);
    Serial.print(" ");
  }
  Serial.println();

  // Die restlichen drei Ziffern zu einer Zahl zusammenfügen
  int number = (data[1] - '0') * 100 + (data[2] - '0') * 10 + (data[3] - '0');
  
  // Verarbeiten des Motors basierend auf der ersten Zahl
  if (data[0] == '1') {  // Servo
    angel = number;
    Serial.print("Servo angle: ");
    Serial.println(angel);
  }
  else if (data[0] == '2') {  // Schrittmotor
    motor_speed = number;
    Serial.print("Motor speed: ");
    Serial.println(motor_speed);
  }
}

void setup() {
  Serial.begin(9600);
  Wire.begin(address);
  Wire.onReceive(receive_data);
  Wire.onRequest(send_response);
}

void loop() {
}
