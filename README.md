# mpar-sim

Multi-function Phased Array Radar (MPAR) Simulator

## Overview

The MPAR simulator is a tool for rapidly simulating radar scenarios at the detection level. The tool is intended to help with rapid prototyping of resource management/tracking algorithms, and the user can easily modify all aspects of the scenario. Targets, waveform parameters, and beam parameters can be modified at runtime.

## Installation

This project can be installed as a python package by running the following in the top-level directory

```bash
pip install -e .
```

## Radar System

The simulation is built around the ```PhasedArrayRadar``` class, which simulates detections for a uniform rectangular array (URA) aperture. A radar can be instantiated as shown below:

```python
radar = PhasedArrayRadar(
    ndim_state=6,
    position_mapping=[0, 2, 4],
    velocity_mapping=[1, 3, 5],
    position=np.array([[0], [0], [0]]),
    rotation_offset=np.array([[0], [0], [0]]),
    # Array parameters
    n_elements_x=32,
    n_elements_y=32,
    element_spacing=0.5,  # Wavelengths
    element_tx_power=10,
    # System parameters
    center_frequency=3e9,
    system_temperature=290,
    noise_figure=4,
    # Scan settings
    beam_shape=SincBeam,
    az_fov=[-45, 45],
    el_fov=[-45, 45],
    # Detection settings
    false_alarm_rate=1e-6,
    include_false_alarms=False
)
```

The ```measure()``` function of the radar can be used to collect measurements for a list of ```GroundTruthState``` objects. These measurements are returned as ```Detection``` objects (```TrueDetection``` if the measurement corresponds to a real target or ```Clutter``` if it's from a false alarm or clutter).

## Features

- Agents:
  - RasterScanAgent
- Beams:
  - RectangularBeam
  - GaussianBeam
  - SincBeam
- Motion models:
  - ConstantVelocity
- Measurement models:
  - CartesianToRangeAzElRangeRate
- Resource management:
  - Best-first scheduler
  - Power-aperture product-based resource manager
- Tracking:
  - Kalman predict
  - Kalman update
  - Extended kalman update
  - Unscented kalman update
