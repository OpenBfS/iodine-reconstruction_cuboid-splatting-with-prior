# Cuboid Splatting-with-prior

This repository contains the official implementation of Cuboid Splatting-with-prior, the method proposed in the paper:

> **Integrating Atmospheric Dispersion Modeling Priors into Cuboid Splatting for Spatiotemporal Reconstruction of Airborne Radioiodine after Nuclear Accidents**  
> for IJCAI-ECAI 2026 35th International Joint Conference on Artificial Intelligence.

Cuboid Splatting-with-prior integrates atmospheric dispersion modeling simulations as informative priors into the path-based reconstruction method Cuboid Splatting, enabling improved spatiotemporal reconstruction of airborne radioiodine concentrations after nuclear accidents.

---

## Repository Structure

The Python modules in this repository provide the following components:

- `pointcloud_init.py`  
  Cuboid Initialization from the prior scenario treated as a pointcloud.

- `supervised_pretraining.py`  
  Supervised pretraining on the prior scenario to obtain the Cuboid Prior.

- `full_finetuning.py`  
  Full finetuning on path data for Cuboid Reconstruction of the spatiotemporal radioiodine concentrations.

- `metrics_for_pipeline.py`  
  Functions to calculate quality metrics (e.g., scenario error and path error).

- `utils_for_pipeline.py`  
  Some helper functions for the overall pipeline.

In addition, the hyperparameter_settings directory contains two CSV files specifying the hyperparameter settings for (i) initialization and pretraining, and (ii) full finetuning, respectively.

---

## Data and Preprocessing

The path data generator and all preprocessing steps for original scenarios and paths, implemented in a previous study by Friedrich et al. (2026) and used in this project, can be downloaded from:

> https://github.com/OpenBfS/iodine-reconstruction

---

## Installation

This project was developed and tested with Python 3.11.8. The requirements.txt covers the requirements for this project as well as the project by Friedrich et al. (2026) to able to run all code in the same environment without version conflicts.

---

## Supplementary Materials

This repository also contains the supplementary materials accompanying the paper:

1. **Extended Experimental Results**  
   Additional results tables from the evaluation experiments.  
   File: [`supplementary_materials/additional_experimental_results.pdf`](supplementary_materials/additional_experimental_results.pdf)

2. **Scenario Descriptions**  
   Details on the original and prior scenarios employed in the evaluation.  
   File: [`supplementary_materials/original_and_prior_scenario_descriptions.pdf`](supplementary_materials/original_and_prior_scenario_descriptions.pdf)
