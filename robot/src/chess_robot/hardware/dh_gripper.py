"""
DH-Robotics PGE-series gripper driver (Modbus RTU over RS485, e.g. via a
Waveshare USB->RS485 converter).

!! VERIFY the register map against the manual shipped with YOUR unit before
hardware use — this is the widely documented PGE map, but firmware revisions
exist. Defaults: 115200 baud, 8N1, slave id 1.

Bring-up CLI (after wiring + 24V):
    python3 -m chess_robot.hardware.dh_gripper --port /dev/ttyUSB0 --test
"""
import time
from enum import IntEnum

REG_INIT = 0x0100         # write 1 = initialize, 0xA5 = full re-initialize
REG_FORCE = 0x0101        # 20..100 (%)
REG_POSITION = 0x0103     # target 0..1000 (permille of stroke; 0=closed, 1000=open)
REG_SPEED = 0x0104        # 1..100 (%)
REG_INIT_STATE = 0x0200   # 0=not initialized, 1=initialized
REG_GRIP_STATE = 0x0201   # see GripState
REG_POS_FB = 0x0202       # current position 0..1000


class GripState(IntEnum):
    MOVING = 0
    ARRIVED_NO_OBJECT = 1
    GRIPPED = 2
    DROPPED = 3


class SimTransport:
    """Simulated gripper for offline tests. `object_at_permille=None` -> empty air."""

    def __init__(self, object_at_permille=None):
        self.object_at = object_at_permille
        self.regs = {REG_INIT_STATE: 0, REG_GRIP_STATE: GripState.ARRIVED_NO_OBJECT,
                     REG_POS_FB: 500, REG_FORCE: 30, REG_SPEED: 50}

    def write(self, reg, value):
        if reg == REG_INIT:
            self.regs[REG_INIT_STATE] = 1
            self.regs[REG_POS_FB] = 1000
        elif reg == REG_POSITION:
            target = int(value)
            if self.object_at is not None and target <= self.object_at:
                self.regs[REG_POS_FB] = self.object_at
                self.regs[REG_GRIP_STATE] = GripState.GRIPPED
            else:
                self.regs[REG_POS_FB] = target
                self.regs[REG_GRIP_STATE] = GripState.ARRIVED_NO_OBJECT
        else:
            self.regs[reg] = int(value)

    def read(self, reg):
        return int(self.regs.get(reg, 0))


class SerialTransport:
    """Real RS485 transport via minimalmodbus (pip3 install minimalmodbus)."""

    def __init__(self, port, slave=1, baudrate=115200, timeout=0.2):
        import minimalmodbus  # lazy
        self.inst = minimalmodbus.Instrument(port, slave)
        self.inst.serial.baudrate = baudrate
        self.inst.serial.timeout = timeout

    def write(self, reg, value):
        self.inst.write_register(reg, int(value), functioncode=6)

    def read(self, reg):
        return int(self.inst.read_register(reg, functioncode=3))


class DHGripper:
    def __init__(self, port='/dev/ttyUSB0', slave=1, baudrate=115200,
                 stroke_mm=26.0, transport=None, logger=None):
        self.stroke_mm = float(stroke_mm)
        self.logger = logger
        self._t = transport
        self._cfg = dict(port=port, slave=slave, baudrate=baudrate)
        self._ready = False

    def connect(self):
        if self._t is None:
            self._t = SerialTransport(self._cfg['port'], self._cfg['slave'],
                                      self._cfg['baudrate'])
        return True

    def initialize(self, full=False, timeout=8.0):
        self.connect()
        self._t.write(REG_INIT, 0xA5 if full else 1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._t.read(REG_INIT_STATE) == 1:
                self._ready = True
                if self.logger:
                    self.logger.info("DH gripper initialized")
                return True
            time.sleep(0.1)
        raise TimeoutError("DH gripper did not report initialized")

    def set_force(self, pct):
        self._t.write(REG_FORCE, max(20, min(100, int(pct))))

    def set_speed(self, pct):
        self._t.write(REG_SPEED, max(1, min(100, int(pct))))

    def move_to(self, permille, wait=True, timeout=3.0):
        self._t.write(REG_POSITION, max(0, min(1000, int(permille))))
        if not wait:
            return GripState.MOVING
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = GripState(self._t.read(REG_GRIP_STATE))
            if st != GripState.MOVING:
                return st
            time.sleep(0.05)
        return GripState.MOVING

    def open(self):
        return self.move_to(1000)

    def close(self, force_pct=None):
        """Close fully; returns GripState. GRIPPED == an object is held."""
        if force_pct is not None:
            self.set_force(force_pct)
        return self.move_to(0)

    def is_holding(self):
        return GripState(self._t.read(REG_GRIP_STATE)) == GripState.GRIPPED

    def position_mm(self):
        return self._t.read(REG_POS_FB) / 1000.0 * self.stroke_mm


def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='/dev/ttyUSB0')
    ap.add_argument('--slave', type=int, default=1)
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    g = DHGripper(port=a.port, slave=a.slave)
    g.initialize()
    if a.test:
        print("open ->", g.open(), "| pos", g.position_mm(), "mm")
        time.sleep(1)
        print("close ->", g.close(force_pct=30), "| pos", g.position_mm(),
              "mm | holding:", g.is_holding())
        g.open()


if __name__ == '__main__':
    _cli()
