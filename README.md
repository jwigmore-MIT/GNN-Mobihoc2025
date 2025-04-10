# MAGNN-QControl: Multi-Class Queueing Network Routing via GNNs

This repository contains the code for training and evaluating Graph Neural Network (GNN)-based routing policies in multi-class multi-hop queueing networks. The core contribution is the **Intranode Aggregation (IA)** mechanism, which facilitates class-level information exchange within nodes while preserving permutation invariance.

## 📂 Repository Structure

### `training/`
- `train.py`: Main training script.
- `run_evaluation.py`: Evaluation driver for trained models.
- `training_utils.py`, `utils.py`: Training loops, PPO implementation, helper functions.

### `configs/`
- JSON files specifying experiment configurations.  
- Example: `Exp2_minQ_spdaIA_bp_EnvSetC.json` uses backpressure routing with SPDA intranode aggregation on Environment Set #1.

### `models/`
- `IntraNodeAggConv.py`: Defines intranode aggregation layers.
- `DeeperIntranodeAggGNN.py`: Full GNN model with message passing and IA.
- `wrappers.py`: Model wrappers for environment interaction.

### `env_sim/`
- Custom TorchRL-compatible simulation environments:
  - `MultiClassMultihopBP.py`: Backpressure-style queueing simulator.
  - `MultiClassMultihopTS.py`: Threshold-style simulator.
- `EnvSampler.py`: Samples environments from a set for training/evaluation.

### `env_creation/`
- Scripts for generating synthetic queueing environments:
  - `create_random_networks.py`: General environment generation logic.
  - `create_ba_context_set_script.py`: Generates **Environment Set #1** (saved in `env_set_c/`).
  - `create_power_util_context_set_script.py`: Generates **Environment Set #2** (saved in `power_util_context_100_set_a/`).
  - `get_network_baselines_and_metrics.py`: Computes metrics and baselines.

### `environment_sets/`
- Pre-generated environment context sets:
  - `env_set_c/`: **Environment Set #1** — referenced in most experiments in the main paper and supplementary material.
  - `power_util_context_100_set_a/`: **Environment Set #2** — used for energy-efficient routing tasks.

### `evaluation/`
- `eval_training_model_mp.py`: Parallel evaluation script for trained policies.

---

## 🚀 Quickstart: Training

To train a model using the same setup as in the paper, run:

```bash
python -m training.train --config configs/<config_file_name>.json
```

For example:

```bash
python -m training.train --config configs/Exp2_minQ_spdaIA_bp_EnvSetC.json/Exp2_minQ_spdaIA_bp_EnvSetC.json
```

- Configs using `env_set_c` → **Environment Set #1**  
- Configs using `power_util_context_100_set_a` → **Environment Set #2**

---

## 📈 Evaluation

Evaluate a trained model using:

```bash
python -m training.run_evaluation --config configs/<config_file_name>.json
```

Or configure `evaluation/eval_training_model_mp.py` for batch evaluation over multiple contexts.

---

## 🧪 Environment Sets (from Paper)

- `env_set_c/` — **Environment Set #1**  
  Diverse Barabási–Albert networks used for training and generalization benchmarks.

- `power_util_context_100_set_a/` — **Environment Set #2**  
  Used for evaluating routing under energy constraints.


## 🛠 Dependencies

This project uses:
- Python 3.8+
- PyTorch
- TorchRL
- NetworkX
- Matplotlib
- TensorDict
- NumPy

Install all requirements with:

```bash
pip install -r requirements.txt
```

---