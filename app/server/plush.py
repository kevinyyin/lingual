"""The plushie: an Arduino on USB (arduino/plush_arms) shows a check and flaps its arms when it reads b"c", an X when it
reads b"x", and a ! ("try again") when it reads b"!". Optional: with no board plugged in, react() does nothing."""

import logging

import serial
from serial.tools import list_ports

ARDUINO_VID = 0x2341  # every official Arduino board, the Uno R4 WiFi included
CODES = {"right": b"c", "wrong": b"x", "retry": b"!"}
log = logging.getLogger("asl.plush")
_port: serial.Serial | None = None


def react(result: str) -> None:
    global _port
    try:
        if _port is None:
            found = next((p.device for p in list_ports.comports() if p.vid == ARDUINO_VID), None)
            if found is None:
                return
            _port = serial.Serial(found, 115200, write_timeout=0)  # never blocks the session's event loop
            log.info("plushie on %s", found)
        _port.write(CODES[result])
    except (serial.SerialException, OSError) as e:  # unplugged mid-lesson: look for it again on the next answer
        log.warning("plushie: %s", e)
        if _port is not None:
            _port.close()
        _port = None
