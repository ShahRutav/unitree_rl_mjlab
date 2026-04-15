#!/usr/bin/env python3
"""
test_hand.py — Standalone Inspire hand test script.

Requires the inspire_g1 service running on the robot:
  sudo ./dfx_inspire_service/build/inspire_g1

Usage
-----
  python3 scripts/test_hand.py                     # auto-detect network interface
  python3 scripts/test_hand.py --interface eth0    # explicit interface
  python3 scripts/test_hand.py --open              # hold open
  python3 scripts/test_hand.py --close             # hold closed
  python3 scripts/test_hand.py --wave              # slow open/close cycle (default)
"""

import argparse
import time

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_


# ---------------------------------------------------------------------------
# Joint labels
# ---------------------------------------------------------------------------

FINGER_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]


# ---------------------------------------------------------------------------
# Hand controller
# ---------------------------------------------------------------------------

class InspireHandTest:
    def __init__(self, interface: str = None):
        if interface:
            ChannelFactoryInitialize(0, interface)
        else:
            # Empty string mirrors C++ Init(0, "") — triggers auto-detection
            ChannelFactoryInitialize(0, "")

        self._cmd = MotorCmds_()
        for _ in range(12):
            self._cmd.cmds.append(unitree_go_msg_dds__MotorCmd_())

        self._pub = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self._pub.Init()

        self._state = None
        self._sub = ChannelSubscriber("rt/inspire/state", MotorStates_)
        self._sub.Init(handler=self._state_cb, queueLen=10)

    def _state_cb(self, msg: MotorStates_):
        self._state = msg

    def send(self, right: np.ndarray, left: np.ndarray) -> None:
        """Send angles. right/left: 6 floats each, range 0.0 (closed) - 1.0 (open)."""
        for i in range(6):
            self._cmd.cmds[i].q     = float(np.clip(right[i], 0.0, 1.0))
        for i in range(6):
            self._cmd.cmds[i + 6].q = float(np.clip(left[i],  0.0, 1.0))
        self._pub.Write(self._cmd)

    def print_state(self):
        if self._state is None:
            print("  (no state received yet)")
            return
        r = [self._state.states[i].q     if self._state.states[i] else 0.0 for i in range(6)]
        l = [self._state.states[i + 6].q if self._state.states[i + 6] else 0.0 for i in range(6)]
        print(f"  right: { {n: f'{v:.2f}' for n, v in zip(FINGER_NAMES, r)} }")
        print(f"  left:  { {n: f'{v:.2f}' for n, v in zip(FINGER_NAMES, l)} }")

    def close(self):
        self._pub.Close()
        self._sub.Close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standalone Inspire hand test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--interface", default=None,
                        help="Network interface for DDS (e.g. eth0). Auto-detected if omitted.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--open",  action="store_true", help="Hold both hands fully open")
    mode.add_argument("--close", action="store_true", help="Hold both hands fully closed")
    mode.add_argument("--wave",  action="store_true", help="Slow open/close cycle (default)")
    args = parser.parse_args()

    print(f"[test_hand] initialising DDS on interface: {args.interface or '(auto)'}")
    hand = InspireHandTest(args.interface)

    # Give the subscriber time to connect
    time.sleep(0.5)

    if args.open:
        print("[test_hand] holding OPEN — Ctrl-C to stop")
        target = np.ones(6, dtype=np.float32)
        try:
            while True:
                hand.send(target, target)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    elif args.close:
        print("[test_hand] holding CLOSED — Ctrl-C to stop")
        target = np.zeros(6, dtype=np.float32)
        try:
            while True:
                hand.send(target, target)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    else:
        # Default: slow open/close wave with live state printout
        print("[test_hand] running open/close wave at 0.2 Hz — Ctrl-C to stop")
        period = 5.0  # seconds per full open→close→open cycle
        t0 = time.monotonic()
        try:
            while True:
                t = time.monotonic() - t0
                # Smooth sinusoidal ramp: 0=closed, 1=open
                val = 0.5 * (1.0 + np.sin(2 * np.pi * t / period - np.pi / 2))
                target = np.full(6, val, dtype=np.float32)
                hand.send(target, target)

                # Print state every 0.5 s
                if int(t * 2) % 1 == 0:
                    print(f"\r[t={t:5.1f}s]  cmd={val:.2f}", end="  ", flush=True)
                    hand.print_state()

                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    print("\n[test_hand] done")
    hand.close()


if __name__ == "__main__":
    main()
