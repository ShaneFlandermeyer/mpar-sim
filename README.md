# mpar-sim

Multi-function Phased Array Radar (MPAR) Simulator

## Overview

The MPAR simulator is a tool for rapidly simulating radar scenarios at the detection level.
The tool is intended to help with rapid prototyping of resource management/tracking algorithms, and the user can easily modify all aspects of the scenario. Targets, waveform parameters, and beam parameters can be modified at runtime.

## Installation

This project can be installed as a python package by running the following in the top-level directory

```bash
pip install -e .
```

## Radar System

The radar system used in this simulator is assumed to be a uniform rectangular array. A radar object can be instantiated as shown below:

```python
radar = PhasedArrayRadar(
      # Platform orientation and position parameters
      position=np.array([0, 0, 0]),
      rotation_offset=StateVector(np.array([0, 0, 0])),
      position_mapping=(0, 2, 4),
      # Array parameters
      n_elements_x=16,
      n_element_y=16,
      element_spacing=0.5, # Wavelengths
      element_tx_power=10,
      # System parameters
      center_frequency=3e9,
      system_temperature=290,
      noise_figure=4,
      # Scan settings
      beam_type=RectangularBeam,
      field_of_view=90,
      # Detection settings
      false_alarm_rate=1e-6,
  )
```

This object maintains information about the physical parameters of the system and can attempt to detect a list of targets using the ```measure()``` function (with measurement probabilities calculated using SNRs from the radar range equation). False alarms are **currently disabled** in this function, but will be added back in after resource management algorithms have been implemented

## Targets and Motion Models

Target and platform maneuvering behavior is handled by the [Stone Soup](https://stonesoup.readthedocs.io/en/latest/index.html) library, which supports a large number of common motion models. See the basic simulator in the ```examples/``` subfolder for an example of how to simulate target motion.

## Resource Management

At each time step of the simulation, one or more radar beams can be transmitted. Relevant parameters are stored in the ```Look``` object, an example of which is shown below.

```python
look = Look(
        start_time=0.1,
        azimuth_steering_angle=10,
        elevation_steering_angle=0,
        azimuth_beamwidth=8,
        elevation_beamwidth=8,
        bandwidth=10e6,
        pulsewidth=10e-6,
        prf=5e3,
        n_pulses=10
    )
```

You can use the ```load_look()``` function in the Radar object to allocate subarrays and update resolutions/radar range equation parameters for this look:

```python
radar.load_look(look)
```

## Existing Agents

- ```RasterScanAgent```: Uniformly scans across a static az/el grid, moving the beam to a new position in the grid at each time step. An example instantiation of this object is given below:

```python
raster_agent = RasterScanAgent(
      azimuth_scan_limits=radar.azimuth_limits,
      elevation_scan_limits=radar.elevation_limits,
      azimuth_beam_spacing=0.85,
      elevation_beam_spacing=0.85,
      azimuth_beamwidth=7.5,
      elevation_beamwidth=7.5,
      bandwidth=5e6,
      pulsewidth=10e-6,
      prf=1500,
      n_pulses=10)
```

## Progress

- [x] Phased array radar detection model
- [x] Multi-target tracking scenario simulation
- [x] Raster scan agent
- [ ] Adaptive Tracking
- [ ] Deterministic task scheduling algorithm(s)
- [ ] False alarm detections
- [ ] Gym environment (for RL)
- [ ] RL-based task scheduler
- [ ] Documentation
