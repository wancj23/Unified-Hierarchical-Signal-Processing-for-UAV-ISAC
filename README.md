# Unified Hierarchical Signal Processing for UAV ISAC

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE TVT](https://img.shields.io/badge/Paper-IEEE_TVT_Under_Review-success.svg)](#)

> **Official Implementation** for the paper:  
> *"Design and Optimization of a Doppler-Resilient ISAC Framework for UAV Hierarchical Sensing"* > Submitted to the **IEEE Transactions on Vehicular Technology (TVT)**.

## 📖 Overview

This repository provides the complete, end-to-end physical layer simulation and signal processing architecture for the proposed **WH-DRCS ISAC Framework**. 

To address the severe Doppler vulnerability and inflexible frame configurations of conventional IEEE 802.11ay/bf standards in high-mobility UAV swarm scenarios, we propose a systematic framework that tightly integrates a novel **Walsh-Hadamard Doppler-Resilient Complementary Sequence (WH-DRCS)** with a suite of **hierarchical signal processing algorithms**. 

This codebase demonstrates the system's capability to seamlessly map raw delay-Doppler echoes into fine-grained semantic perceptions across four discrete hierarchical levels ($L_1$ to $L_4$), achieving a Pareto-optimal trade-off between Gbps-level communication and robust micro-Doppler kinematic tracking.

### 🌟 Core Algorithmic Mechanisms Highlighted

- **Mechanism 1 (Nyquist-Folded Micro-Doppler Extraction):** Located in `src/l2_target_classification.py`. Utilizes boundary truncation ratios to distinguish highly dynamic UAV rotors from generic moving scatterers.
- **Mechanism 2 (Adaptive Density-Based Kinematic Clustering):** Located in `src/l3_configuration_recognition.py`. Employs dynamic $\epsilon$-scaling DBSCAN to accurately isolate discrete rotor signatures and map them directly to physical UAV topologies.

------

## 📂 Repository Structure

```text
Unified-Hierarchical-Signal-Processing-for-UAV-ISAC/
├── data/                               # Sample/Generated data directories
│   └── rda.json                        # Sample discrete delay-Doppler matrix output
├── scripts/                            # Automated execution scripts
│   ├── 01_generate_channel.sh          # Synthesizes dynamic UAV channels (NIST QD)
│   ├── 02_run_isac_plm.sh              # Executes WH-DRCS baseband simulation (ISAC-PLM)
│   └── 03_evaluate_perception.sh       # Runs the L1-L4 semantic perception pipeline
├── simulation_env/                     # Underlying physical-layer simulation models
│   ├── isac-plm/                       # Modified NIST ISAC-PLM (with WH-DRCS injected)
│   └── nist-qd/                        # NIST Quasi-Deterministic Channel Model
├── src/                                # Core Hierarchical Perception Source Code
│   ├── unified_hierarchical_processor.py # Master pipeline orchestrator
│   ├── l1_fundamental_detection.py     # Stage 1: CFAR & Kinematic Entropy
│   ├── l2_target_classification.py     # Stage 2: Target Discrimination (Mechanism 1)
│   ├── l3_configuration_recognition.py # Stage 3: Topology Mapping (Mechanism 2)
│   └── l4_attitude_estimation.py       # Stage 4: 3D Spatial Attitude Inversion
├── requirements.txt                    # Python environment dependencies
└── README.md
```

## 🚀 Quick Start & Reproducibility

We have provided automated bash scripts to ensure the rigorous reproducibility of our system-level evaluations.

### 1. Environment Setup

Ensure you have Python 3.8+ installed. Install the required numerical and machine-learning dependencies:

```
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline

Execute the following scripts sequentially to simulate the channel, generate the baseband radar echoes, and perform the multi-tiered semantic perception:

```
# Step 1: Generate the multi-path micro-Doppler channel matrix
bash scripts/01_generate_channel.sh

# Step 2: Modulate the WH-DRCS waveform and extract the delay-Doppler matrix
bash scripts/02_run_isac_plm.sh

# Step 3: Map the raw baseband data into L1-L4 hierarchical semantic outputs
bash scripts/03_evaluate_perception.sh
```

*(Note: Step 3 can be run standalone using the provided data/rda.json sample file to evaluate the algorithmic mechanisms directly).*

------

## 📜 Acknowledgments & License Statement

This framework builds upon the foundational physical layer and channel models provided by the **National Institute of Standards and Technology (NIST)**.

- **NIST ISAC-PLM:** The base IEEE 802.11ay/bf EDMG simulation environment.
- **Modifications:** In compliance with the NIST licensing terms, we explicitly acknowledge that the `simulation_env/isac-plm/` source code has been significantly modified by our team. Specifically, we developed and injected the **WH-DRCS sequence generation, deterministic orthogonal masking, and scalable ZAZ framing modules** to replace the standard Golay complementary sequence implementations.

The hierarchical perception algorithms (`src/` directory) are originally developed by the authors of this paper.

------

## 📬 Contact

For any queries regarding the codebase, mathematical proofs, or dataset access, please reach out to the author

**Chengjuan Wan** University of Chinese Academy of Sciences, Beijing, P.R.China

Email: `wanchengjuan23@mails.ucas.ac.cn`

