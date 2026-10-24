"""The plushie's arms: an Arduino on USB (arduino/plush_arms) flaps them when it reads b"c". Optional: with no board
plugged in, cheer() does nothing."""

import logging

import serial
from serial.tools import list_ports

ARDUINO_VID = 0x2341  # every official Arduino board, the Uno R4 WiFi included
log = logging.getLogger("asl.plush")
_port: serial.Serial | None = None


def cheer() -> None:
    global _port
    try:
        if _port is None:
            found = next((p.device for p in list_ports.comports() if p.vid == ARDUINO_VID), None)
            if found is None:
                return
            _port = serial.Serial(found, 115200, write_timeout=0)  # never blocks the session's event loop
            log.info("plushie arms on %s", found)
        _port.write(b"c")
    except (serial.SerialException, OSError) as e:  # unplugged mid-lesson: look for it again on the next cheer
        log.warning("plushie arms: %s", e)
        if _port is not None:
            _port.close()
        _port = None
