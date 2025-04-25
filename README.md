# Ring-AllReduce Simulation Environment

This project is part of my Final Year Project module at UCL. This is a Ring-AllReduce Simulation Environment to explore the performance of compression schemes in a Distributed Deep Learning setting.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

---

## Prerequisites

- **Python** >=3.13
- pip (usually bundled with Python)

---

## Installation

1. Clone the repo  
   ```bash
   git clone https://github.com/QErywan/RARSimulation.git
   cd RARSimulation

2. Create a Virtual Environment
    ```bash
    python3 -m venv .venv


3. Activate the Virtual Environment
    ```bash
    source .venv/bin/activate

4. Upgrade pip
    ```bash
    pip install --upgrade pip

5. Install project requirements
    ```bash
    pip install -r requirements.txt


## Usage

### Changing Parameters

Parameters can be found in `ringallreduce_sim.py`

### Simulation Parameters

| Parameter             | Description                                                                                                   |
|-----------------------|---------------------------------------------------------------------------------------------------------------|
| `seed`                | Random seed for reproducibility.                                                                              |
| `gpu`                 | GPU ID to use (0 for first GPU, -1 for CPU).                                                                  |
| `num_rounds`          | Number of communication rounds in the simulation.                                                             |
| `num_clients`         | Number of participating clients.                                                                              |
| `test_every`          | Frequency (in rounds) to perform testing.                                                                     |
| `lr`                  | Learning rate used during training.                                                                           |
| `lr_type`             | Type of learning rate scheduling. Options: `'const'`, `'step_decay'`, `'exp_decay'`.                         |
| `client_train_steps`  | Number of local training steps per client before communication.                                               |
| `client_batch_size`   | Batch size used by each client (for both training and testing).                                               |
| `net`                 | CNN model architecture to use, e.g. `'ResNet9'`.                                                              |
| `dataset`             | Dataset to use, e.g. `'CIFAR10'`.                                                                             |
| `error_feedback`      | Toggle for error feedback mechanism (currently not implemented).                                              |
| `nbits`               | Number of bits per coordinate used in the compression scheme.                                                 |
| `compression_scheme`  | Compression/decompression strategy. Options: `'none'`, `'vector_topk'`, `'chunk_topk_recompress'`, `'chunk_topk_single'`, `'csh'`, `'cshtopk_actual'`, `'cshtopk_estimate'`. |
| `sketch_col`          | Number of columns in the sketch matrix (used in certain compression schemes).                                |
| `sketch_row`          | Number of rows in the sketch matrix.                                                                          |
| `k`                   | Top-K value (used in compression schemes like Top-K or CSH-TopK).                                             |
| `data_per_client`     | Data distribution method across clients. Options: `'sequential'`, `'label_per_client'`.                      |
| `folder`              | Folder name where results will be saved.                                                                     |


### Running Simulation

1. To run the simulation:
    ```bash
    python3 ringallreduce_sim.py

