# mpar-sim

Multi-function Phased Array Radar (MPAR) Simulator

## Overview

The MPAR simulator is a python-based simulation tool that I developed as part of my Master's thesis, and am actively maintaining as I continue my Ph.D. research. The repo currently has many tools for radar scenario simulation and tracking. Unfortunately for everyone involved, I don't get paid enough to document this thoroughly. However, most of the major classes have unit tests in the ```test/``` subdirectory that show examples of how to use each class.

In the future, I would like to add some I/Q level simulation, probably GPU accelerated with Jax.

## Features

- Phased array radar detection simulation
- Tracking
  - Kalman filters (KF, EKF, UKF)
  - IMM Estimator
  - PDA/JPDA
  - Global Nearest Neighbor
    - Auction algorithm
- Motion models:
  - Nearly constant velocity (NCV)
  - Nearly constant acceleration (NCA)
  - Constant turn rate (CTR)
- Measurement models:
  - Linear
  - Cartesian to azimuth/elevation/range/velocity
- RCS Models:
  - Swerling 0-3
- And more!

## Installation

This project can be installed as a python package by running the following in the top-level directory

```bash
pip install -e .
```