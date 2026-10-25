// The plushie reacts to teaching-mode answers sent by the server over USB (server/plush.py):
//   'c' right: a check on the neck's dot matrix, the right LED on, and the arms flap
//   'x' wrong: an X on the matrix and the wrong LED on
//   '!' try again (a signing attempt missed, there's another go): a ! on the matrix
//
// Finding the arm angles (server stopped, Serial Monitor at 115200 baud): "a90" turns arm A (pin 6) to 90 degrees,
// "b90" turns arm B (pin 9). For each arm, find where it hangs down and where it's raised without grinding.
#include <Servo.h>

const int A_PIN = 6, A_DOWN = 20, A_UP = 160;
const int B_PIN = 9, B_DOWN = 160, B_UP = 20;
const int STEP_MS = 1;   // ms per 1% of the swing; 1 asks for more than a 9g servo can do, so it runs flat out
const int HOLD_MS = 150; // pause at the top and bottom, so the arm gets all the way there before turning back
const int FLAPS = 3;     // up-and-downs per right answer

const int RIGHT_LED = 2, WRONG_LED = 3;
const int MX_DIN = 11, MX_CS = 10, MX_CLK = 13;  // MAX7219 8x8 dot matrix
const int BRIGHTNESS = 3;    // 0-15: the servos share the USB power, keep it low
const bool MIRROR = false;   // true if the check comes out backwards
const int SHOW_MS = 1500;    // how long the X or ! stays up (the check stays while the arms flap)

const byte CHECK[8] = {0x00, 0x01, 0x03, 0x06, 0x8C, 0xD8, 0x70, 0x20};
const byte CROSS[8] = {0x81, 0xC3, 0x66, 0x3C, 0x3C, 0x66, 0xC3, 0x81};
const byte BANG[8] = {0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x18};

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

void matrix(byte reg, byte val) {
  digitalWrite(MX_CS, LOW);
  shiftOut(MX_DIN, MX_CLK, MSBFIRST, reg);
  shiftOut(MX_DIN, MX_CLK, MSBFIRST, val);
  digitalWrite(MX_CS, HIGH);
}

byte mirrored(byte b) {
  byte r = 0;
  for (int i = 0; i < 8; i++) if (b & (1 << i)) r |= 0x80 >> i;
  return r;
}

void show(const byte *rows) {  // nullptr clears the matrix
  for (int r = 0; r < 8; r++) matrix(r + 1, rows ? (MIRROR ? mirrored(rows[r]) : rows[r]) : 0);
}

void react(const byte *symbol, int led, bool flapArms) {  // led -1: none
  show(symbol);
  if (led >= 0) digitalWrite(led, HIGH);
  if (flapArms) flap(); else delay(SHOW_MS);
  if (led >= 0) digitalWrite(led, LOW);
  show(nullptr);
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(3000);  // "a90" typed key by key: a slow typist must not send the arm to 0
  pinMode(RIGHT_LED, OUTPUT);
  pinMode(WRONG_LED, OUTPUT);
  pinMode(MX_DIN, OUTPUT);
  pinMode(MX_CLK, OUTPUT);
  pinMode(MX_CS, OUTPUT);
  digitalWrite(MX_CS, HIGH);
  matrix(0x0F, 0);           // display test off
  matrix(0x09, 0);           // raw pixels, no digit decoding
  matrix(0x0B, 7);           // all 8 rows
  matrix(0x0A, BRIGHTNESS);
  matrix(0x0C, 1);           // wake up
  show(nullptr);
  rest();
}

void loop() {
  int cmd = Serial.read();
  if (cmd == 'c') react(CHECK, RIGHT_LED, true);
  else if (cmd == 'x') react(CROSS, WRONG_LED, false);
  else if (cmd == '!') react(BANG, -1, false);
  else if (cmd == 'a') turn(armA, A_PIN, Serial.parseInt());
  else if (cmd == 'b') turn(armB, B_PIN, Serial.parseInt());
}
