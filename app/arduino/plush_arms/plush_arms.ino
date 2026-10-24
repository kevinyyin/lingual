// The plushie's arms: two 9g servos (pins 6 and 9) flap when the server sends 'c' over USB (server/plush.py),
// i.e. when the child gets an answer right in teaching mode.
#include <Servo.h>

// Tune these on the real plush: arms resting and arms raised.
// If a servo buzzes at either end, it is straining against the plush: move that number back by 5-10.
const int DOWN = 20;
const int UP = 110;
const int STEP_MS = 2;  // ms per degree; 2 is about as fast as a 9g servo turns, bigger = slower
const int FLAPS = 3;    // up-and-downs per right answer

Servo armA, armB;

void setArms(int angle) {
  armA.write(angle);
  armB.write(180 - angle);  // servos sit mirrored on opposite sides, so one turns the other way
}

void flap() {
  for (int i = 0; i < FLAPS; i++) {
    for (int a = DOWN; a <= UP; a++) { setArms(a); delay(STEP_MS); }
    delay(80);
    for (int a = UP; a >= DOWN; a--) { setArms(a); delay(STEP_MS); }
    delay(80);
  }
}

void setup() {
  Serial.begin(115200);
  armA.attach(6);
  armB.attach(9);
  setArms(DOWN);
}

void loop() {
  if (Serial.read() == 'c') flap();
}
