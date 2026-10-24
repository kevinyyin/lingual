// The plushie's arms: two 9g servos flap when the server sends 'c' over USB (server/plush.py),
// i.e. when the child gets an answer right in teaching mode.
//
// Finding the angles (server stopped, Serial Monitor at 115200 baud): "a90" turns arm A (pin 6) to 90 degrees,
// "b90" turns arm B (pin 9). For each arm, find where it hangs down and where it's raised without grinding.
#include <Servo.h>

const int A_PIN = 6, A_DOWN = 20, A_UP = 160;
const int B_PIN = 9, B_DOWN = 160, B_UP = 20;
const int STEP_MS = 1;   // ms per 1% of the swing; 1 asks for more than a 9g servo can do, so it runs flat out
const int HOLD_MS = 150; // pause at the top and bottom, so the arm gets all the way there before turning back
const int FLAPS = 3;     // up-and-downs per right answer

Servo armA, armB;

void turn(Servo &arm, int pin, int angle) {
  if (!arm.attached()) arm.attach(pin);  // attaching twice takes a second timer slot that keeps driving the pin
  arm.write(angle);
}

void setArms(int pct) {  // 0 = down, 100 = up
  turn(armA, A_PIN, map(pct, 0, 100, A_DOWN, A_UP));
  turn(armB, B_PIN, map(pct, 0, 100, B_DOWN, B_UP));
}

// Let go once down: a powered servo held against the plush grinds (and heats up) for as long as it's held.
void rest() {
  setArms(0);
  delay(300);
  armA.detach();
  armB.detach();
}

void flap() {
  for (int i = 0; i < FLAPS; i++) {
    for (int p = 0; p <= 100; p++) { setArms(p); delay(STEP_MS); }
    delay(HOLD_MS);
    for (int p = 100; p >= 0; p--) { setArms(p); delay(STEP_MS); }
    delay(HOLD_MS);
  }
  rest();
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(3000);  // "a90" typed key by key: a slow typist must not send the arm to 0
  rest();
}

void loop() {
  int cmd = Serial.read();
  if (cmd == 'c') flap();
  else if (cmd == 'a') turn(armA, A_PIN, Serial.parseInt());
  else if (cmd == 'b') turn(armB, B_PIN, Serial.parseInt());
}
