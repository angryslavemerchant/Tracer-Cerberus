# Motor Driver Firmware Changes

## Goal

The vision loop on the Pi needs to know the motor's **actual current angle** after each
command, not just the angle it was told to go to. This closes the position loop properly.

---

## What to Add

After the motor driver processes an incoming movement packet, echo back the actual shaft
angles over serial in a parseable format.

### Suggested response format

```
A{tilt_angle:.4f}P{pan_angle:.4f}\n
```

Example: `AT0.1234P-0.5678\n`

This mirrors the command format (`T...P...`) but prefixed with `A` (actual) so the Pi
can distinguish it from other serial output.

### Where to add it in SimpleFOC

After you parse the incoming packet and call `motor.move(target_angle)` (or equivalent),
add:

```cpp
// echo actual shaft angles back to Pi
Serial.print("AT");
Serial.print(tilt_motor.shaft_angle, 4);
Serial.print("P");
Serial.println(pan_motor.shaft_angle, 4);
```

The Pi reads this after every packet it sends, so timing matters — make sure this line
runs *after* the move command, not before.

---

## Timing expectation

The Pi sends packets at 20Hz (every 50ms). It will read back whatever is in the serial
buffer after sending. The response doesn't need to be instantaneous — the Pi will take
the most recently received angle each tick.

---

## Checklist

- [ ] Identify where incoming packets are parsed in the firmware
- [ ] Add the Serial echo after `motor.move()` / angle command
- [ ] Flash and test with a serial monitor — move the motor by hand and confirm angles
      are printing continuously / on command
- [ ] Verify pan and tilt are in the correct order in the response (matches command order)
- [ ] Confirm angle units are radians (SimpleFOC default is radians)
