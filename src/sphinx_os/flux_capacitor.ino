// Flux Capacitor Arduino Control Code
int signalPin = 9; // PWM output pin for electromagnet control

void setup() {
  pinMode(signalPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    int signalValue = data.toInt();
    analogWrite(signalPin, signalValue);
  }
}
