from types import SimpleNamespace

import serial

from server import plush


class FakePort:
    def __init__(self, device, *_, **__):
        self.device, self.sent, self.unplugged = device, [], False

    def write(self, b):
        if self.unplugged:
            raise serial.SerialException("device disconnected")
        self.sent.append(b)

    def close(self):
        pass


def test_cheer_finds_the_arduino_and_survives_unplugging(monkeypatch):
    devices, opened = [], []
    monkeypatch.setattr(plush.list_ports, "comports", lambda: devices)
    monkeypatch.setattr(plush.serial, "Serial", lambda *a, **k: opened.append(FakePort(*a, **k)) or opened[-1])
    monkeypatch.setattr(plush, "_port", None)

    plush.cheer()  # no board: nothing happens
    assert opened == []

    devices += [SimpleNamespace(device="/dev/cu.Bluetooth", vid=None), SimpleNamespace(device="/dev/cu.usbmodem1", vid=0x2341)]
    plush.cheer()
    plush.cheer()
    assert [p.device for p in opened] == ["/dev/cu.usbmodem1"] and opened[0].sent == [b"c", b"c"]

    opened[0].unplugged = True
    plush.cheer()  # lost: logged and dropped
    plush.cheer()  # plugged back in: opened again
    assert len(opened) == 2 and opened[1].sent == [b"c"]
