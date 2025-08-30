# Affordance-GraspNeRF-Integration

This project was developed for research purposes, with the goal of integrating an Affordance Learning–based model with a NeRF-based model for robotic grasping applications.

## Requirements

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)  
- Python 3.9 and Python 3.11 (or just Miniconda to manage environments)

Two separate virtual environments are required:

- A Python 3.11 environment for [VRB](https://github.com/shikharbahl/vrb.git)  
- A Python 3.9 environment for [GraspNeRF](https://github.com/PKU-EPIC/GraspNeRF.git)  

Follow the installation instructions provided in each project repository to set up the environments.

## Instructions

This repository provides the project structure and the modified files required for integration:

- For **VRB**, the `inference.py` file included here will overwrite the original.  
- For **GraspNeRF**, the following files overwrite the originals:
  - `run_simgrasp.sh`
  - `sim_grasp.py`
  - `stat_result.py`
  - `main.py`
  - `render.py`

These files are based on the original GraspNeRF and VRB code, manyly remained almost completely the same, but include **essential modifications** for enabling integration.  
In particular, `main.py` (inside the `__call__` function) was updated so that the Quality Grasp Volume is boosted using affordance contact points inferred via VRB.  

For the remaining files, simply mirror the structure provided in this repository when placing them.

### Running a Simulation

Before running any configuration:

1. Open `main.py` and check the `__call__` function.  
   Comment or uncomment the relevant code to enable:
   - Baseline GraspNeRF only  
   - Integration with VRB (single-view or multi-view variations)

2. Ensure that the configuration in `main.py` matches the settings in either:
   - `run_simgrasp.sh`  
   - `server_simgrasp.sh`  

Running either script will start a simulation.

## Data & Models (Coming Soon)

This project requires additional assets that are **not yet available** in this repository:

- Dataset used for testing, base model and integrations [Dataset](https://drive.google.com/drive/folders/1fHIzFR7_ZlyhGygTCff_nizSWnCUeJ65?usp=sharing).
- VRB model checkpoint.
- GraspNeRF assets.

These resources will be uploaded and linked here shortly.

## Citations

This project builds upon the following works:

- **GraspNeRF**  
  Qiyu Dai, Yan Zhu, Yiran Geng, Ciyu Ruan, Jiazhao Zhang, He Wang (2023).  
  *GraspNeRF: Multiview-based 6-DoF Grasp Detection for Transparent and Specular Objects Using Generalizable NeRF*.  
  [arXiv Paper](https://arxiv.org/abs/2210.06575) | [GitHub Repository](https://github.com/PKU-EPIC/GraspNeRF.git)

- **VRB (Vision-Robotics Bridge)**  
  Shikhar Bahl, Russell Mendonca, Lili Chen, Unnat Jain, Deepak Pathak (2023).  
  *VRB: Affordances from Human Videos as a Versatile Representation for Robotics*.  
  [arXiv Paper](https://arxiv.org/abs/2304.08488) | [GitHub Repository](https://github.com/shikharbahl/vrb.git)

