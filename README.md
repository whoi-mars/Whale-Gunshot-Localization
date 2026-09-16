# Automatic Detection and Localization of an Unknown Number of Acoustic Sources

A research framework for underwater acoustic source localization using machine learning and multi-sensor signal processing.

This repository contains tools for simulating underwater acoustic recordings, training deep learning models for source localization and detection, associating detections across hydrophones, and evaluating localization performance on both simulated and experimental datasets.

The code was developed to investigate passive acoustic localization of impulsive sources such as airgun shots and whale vocalizations using distributed hydrophone arrays.

---

## Overview

The repository combines several components into a single workflow:

- Simulation of underwater acoustic datasets
- Deep learning models for localization and classification
- Multi-sensor data association
- Source localization and tracking algorithms
- Evaluation on simulated and experimental data

Together these components provide an end-to-end framework for developing and testing passive acoustic localization methods.

---

## Repository Structure

```
Whale-Gunshot-Localization/
│
├── whale_gunshot_localization/
│   Core Python package
│   ├── datasets/
│   ├── losses/
│   ├── models/
│   ├── sim_tools/
│   └── utils/
│
├── train/
│   Training scripts for localization and classification models
│
├── associate_and_localize/
│   Data association and localization algorithms
│
├── evaluate/
│   Evaluation notebooks and experimental analyses
│
├── experiments/
│   Simulation studies and benchmarking experiments
│
├── matlab/
│   MATLAB utilities used for simulation and analysis
│
├── scripts/
│   Helper scripts
│
└── tests/
│   Unit tests
```

---

## Features

### Machine Learning

- Fully convolutional and Temporal Convolutional Network (TCN) models
- Joint localization and classification tasks
- Custom loss functions for localization and multitask learning
- Configurable training pipelines

### Simulation

- Synthetic underwater acoustic data generation
- Acoustic propagation simulation utilities
- Dataset generation notebooks

### Localization

- Multi-sensor data association
- Source localization algorithms
- Source tracking experiments
- Missing-data tracking simulations

### Evaluation

- Localization accuracy evaluation
- Range estimation experiments
- Experimental data analysis
- Visualization notebooks

---

## Installation

Clone the repository

```bash
git clone https://github.com/whoi-mars/Whale-Gunshot-Localization.git
cd Whale-Gunshot-Localization
```

Install the required Python packages

```bash
pip install -r requirements.txt
```

---

## Configuration

Experiment parameters are specified using YAML configuration files located in

```
whale_gunshot_localization/
```

Several example configurations are included for different experimental setups.

---

## Typical Workflow

### 1. Generate a dataset

Simulation notebooks are provided for generating training datasets from synthetic acoustic propagation simulations.

Examples include

- `create_dataset_RD.ipynb`
- `create_dataset_RI.ipynb`

---

### 2. Train a model

Training scripts are located in

```
train/
```

Model architecture and training parameters are controlled through the configuration files.

---

### 3. Evaluate performance

The `evaluate/` directory contains notebooks for

- localization accuracy
- range estimation
- experimental data analysis
- source association
- visualization

These notebooks reproduce many of the analyses used throughout the project.

---

### 4. Associate detections and localize sources

The `associate_and_localize/` directory contains algorithms for combining detections across multiple hydrophones to estimate source locations.

These methods are designed to operate on model predictions and demonstrate complete localization pipelines beyond individual detection models.

---

## Models

The repository currently includes several neural network architectures, including

- Temporal Convolutional Networks (TCNs)
- Fully Convolutional Networks (FCNs)
- LeNet-based models

These models are implemented within

```
whale_gunshot_localization/models/
```

and can be trained using the provided training scripts.

---

## Package Organization

### datasets/

Dataset loading and PyTorch dataloaders.

### models/

Neural network implementations.

### losses/

Custom localization, classification, and multitask loss functions.

### sim_tools/

Simulation utilities for generating synthetic acoustic datasets.

### utils/

Supporting utilities for transformations, graph algorithms, mathematical routines, and experimental analyses.

---

## Experiments

The repository includes reproducible simulation studies investigating topics such as

- source tracking
- missing detections
- source separation
- localization performance
- computational timing

Results from these experiments are stored in the `experiments/results/` directory.

---

## Citation

If you use this repository in your research, please cite the following.
```bibtex
@article{11402918,
   author={Goldwater, Mark and Bonnel, Julien and McNeese, Andrew R. and Wilson, Preston S. and Zitterbart, Daniel P.},
   journal={IEEE Journal of Oceanic Engineering}, 
   title={Automatic Detection and Localization of an Unknown Number of Acoustic Sources Using a Network of Unsynchronized Hydrophones in a Dispersive Waveguide}, 
   year={2026},
   volume={51},
   number={2},
   pages={894-915}
}
```
