# Gimbal Control — Design Notes

## Overview

This document covers the design and implementation of the gimbal control pipeline
added in `main_gimbal.py`. It runs alongside the Tracer-Cerberus vision tracker and
drives a two-axis pan/tilt gimbal via a serial-connected motor driver.

---

## Hardware

- **Pan motor:** rotates the entire assembly. Tilt axis centre is 63.5mm from pan axis.
- **Tilt motor:** rotates the camera. Camera sensor is 30mm from tilt axis.
- **Camera:** Raspberry Pi Camera Module 3 Noir (imx708), 1280x720, focal length ~940px.
- **Motor driver:** custom SimpleFOC-based controller over UART at 115200 baud.

---

## Pipeline

```
camera frame
  -> tracker.update()         -> error_x, error_y (pixels from frame centre)
  -> pixel_to_angles()        -> alpha_x, alpha_y (radians, camera frame)
  -> EMA filter               -> smoothed_x, smoothed_y
  -> PD controller            -> pid_x, pid_y
  -> angles_to_motor_delta()  -> delta_pan, delta_tilt (geometric correction)
  -> actual + delta           -> absolute target angles
  -> serial packet            -> motor driver
```

---

## Coordinate System

- `error_x` positive = target is right of frame centre -> pan right (positive pan)
- `error_y` positive = target is below frame centre -> tilt down (negative tilt)
- Motor driver receives absolute angles in radians from its zero position (boot position)
- Both sides share the same zero: offsets are subtracted in firmware before echoing back

---

## Mount Geometry Correction

The camera is not coaxially mounted on either axis. When a motor moves, the camera
both rotates AND translates, amplifying the effective pointing motion at close range.
A naive angle command would undershoot — we correct with:

```
k = 1 / (1 + offset / distance)
```

- **Pan:** effective radius = `63.5 + 30 * cos(tilt)` mm (varies with tilt angle)
- **Tilt:** fixed radius = 30mm

At 300mm working distance, pan correction is ~24% (k ≈ 0.76). At 1500mm it's ~6%.
Distance is estimated from the YOLO bounding box height at acquisition and locked for
the life of the track:

```
d_mm = (TARGET_HEIGHT_MM * FOCAL_LENGTH_PX) / bbox_h_px
```

---

## PD Controller

```
smoothed = EMA_ALPHA * raw + (1 - EMA_ALPHA) * smoothed   # noise filter
d        = (smoothed - prev_smoothed) / dt                 # derivative
output   = K_P * smoothed + K_D * d
```

The EMA filter smooths the noisy heatmap-based error signal before differentiation.
Without it, the D term amplifies tracker noise into violent motor commands.

On lock transition (SEARCHING -> TRACKING), the EMA state is seeded with the current
error so the derivative starts at zero rather than spiking.

### Tuned values

| Parameter    | Pan  | Tilt |
|---|---|---|
| K_P          | 1.0  | 1.0  |
| K_D          | 0.08 | 0.05 |
| EMA_ALPHA    | 0.4  | 0.4  |

K_P = 1.0 means "command the full geometric correction each tick." This works cleanly
because the feedback loop is closed — the motor's own position PID handles smoothness,
and any overshoot is corrected on the next tick from real shaft angle feedback.

The I term is implemented but zeroed out. With clean closed-loop feedback and no
significant friction load, there is no persistent steady-state error for I to correct.

---

## Closed-Loop Feedback

The motor driver echoes actual shaft angles after every command:

```
AT{tilt:.4f}P{pan:.4f}\n   (angles relative to zero, radians)
```

The Pi parses the most recent response each control tick and uses it as the base for
the next correction:

```
target = actual_angle + geometric_correction(pid_output)
```

This replaces the previous dead-reckoning approach which accumulated commanded angles
in software. Dead reckoning drifted because motor overshoot, friction, and inertia meant
the actual position never matched the commanded position — errors compounded silently.
With shaft angle feedback, all disturbances are corrected automatically on the next tick.

---

## Motor Driver Protocol

| Command | Description |
|---|---|
| `PING\n` | Ping — driver responds `READY` |
| `ACK\n` | Acknowledge READY |
| `ENABLE\n` | Enable motor output |
| `DISABLE\n` | Disable motor output |
| `T{t:.4f}P{p:.4f}:{checksum}\n` | Move to absolute tilt/pan angles (radians) |

Checksum: sum of ASCII values of the data string, masked to 8 bits.

Driver echoes `AT{tilt:.4f}P{pan:.4f}\n` after each valid movement packet.

---

## Key Controls (display window)

| Key | Action |
|---|---|
| `g` | Resume sending motor commands |
| `s` | Send zero-zero, stop motor commands (display and tracking continue) |
| `r` | Reset tracker to SEARCHING |
| `q` | Quit |

---

## Tuning Guide

1. **Start with K_P only, K_D = 0.** Raise K_P until tracking is responsive.
2. **If oscillating:** lower K_P, or raise EMA_ALPHA toward 0.6 to smooth more.
3. **Add K_D** once P is stable. Raises snap without oscillation by backing off as
   the motor approaches target. If D causes jerk, EMA_ALPHA is too high (not enough
   smoothing) or K_D is too large.
4. **Add K_I** only if there is a consistent steady-state offset that P+D cannot correct
   (e.g. heavy load on one axis causing the motor to consistently undershoot).
5. Pan and tilt need independent tuning — pan has significantly more inertia.
