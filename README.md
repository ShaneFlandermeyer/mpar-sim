# mpar-sim

Multi-function Phased Array Radar (MPAR) Simulator

## Overview

The MPAR simulator is a tool for rapidly simulating radar scenarios at the detection level.
The tool is intended to help with rapid prototyping of resource management/tracking algorithms, and the user can
easily modify all aspects of the scenario. Target trajectories can be set at the beginning of the simulation, while waveform/beam parameters can be modified at runtime.

## Radar System

The radar system used in this simulator is assumed to be a uniform rectangular array positioned at the origin of the scenario coordinate system. A radar object can be instantiated as shown below:

```python
radar = Radar(
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
      azimuth_limits=np.array([-60, 60]),
      elevation_limits=np.array([-20, 20]),
      # Detection settings
      false_alarm_rate=1e-6,
  )
```

This object maintains information about the physical parameters of the system, along with the subarray occupancy at a given time step. Internally, the radar has a ```RadarDetectionGenerator``` object that generates a list of detections given the targets in the scenario and the current beam allocations (using the radar range equation to compute detection probabilities and SNRs).

## Targets and Motion Models

The ```Platform``` object can be used to model targets in the environment. These targets are currently limited to constant position/velocity/acceleration/RCS, but more advanced target models are planned in a future release.

Multi-target scenarios can be simulated by maintaining a list of targets in the scenario. For example, 50 targets with with uniformly distributed velocity coordinates and $RCS = 10\ m^2$ could be placed at the origin as follows:

```python
targets = [Platform(position=np.zeros((3,)),
                    velocity=np.random.rand(3), 
                    rcs=10) for _ in range(50)]
```

With a motion profile defined, target trajectories can be stepped through time using the ```update()``` function:

```python
for target in self.targets:
      target.update(dt=action.dwell_time)
```

In a future release, ```Platform``` objects will be used to handle radar system motion as well.

## Beams

The MPAR simulator accounts for changes in beamwidth and directivity due to scanning off boresight, along with gains/losses due to the shape of the beam. The following beam shapes are currently supported:

- Rectangular beam: Full power inside the az/el beamwidth, zero elsewhere
- Gaussian beam with no sidelobes

## Resource Management

At each time step of the simulation, one or more radar beams can be transmitted. Relevant parameters are stored in the ```RadarLook``` object, an example of which is shown below.

```python
look = RadarLook(
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