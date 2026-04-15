PathFinder: Advancing Path Loss Prediction for Single-to-Multi-Transmitter Scenario
=================================================================================

This repository contains the official implementation of the paper
"PathFinder: Advancing Path Loss Prediction for Single-to-Multi-Transmitter Scenario". 

The paper has been accepted by **Pattern Recognition**.

Project page: https://emorzz1g.github.io/PathFinder

Dynamic Demonstrations
--------
<p align="center">
<img src="docs/gifs/main_trial1_v1_walk_11.gif" width="32%" />
<img src="docs/gifs/main_trial1_v1_walk_450.gif" width="32%" />
<img src="docs/gifs/main_trial1_v1_walk_800.gif" width="32%" />
<em>Real-time path loss prediction with a walking transmitter across different urban layouts.</em>
</p>

Highlights
----------
- Multi-GPU training and inference via PyTorch DDP.
- Config-driven experiments (YAML).
- Baselines and ablations supported by swapping config or entry scripts.

Requirements
------------
- Python 3.8+ (tested with PyTorch)
- CUDA-capable GPU(s) for training

Install dependencies:

```bash
pip install -r requirements.txt
```

Project Structure
-----------------
- config/: YAML experiment configs
- models/: model code
- logs/: training logs (created at runtime)
- results/: inference outputs (created at runtime)
- trainer/: training and inference logic

Pre-trained Model Link
----
You can download our pre-trianed models from Google Driver.

https://drive.google.com/drive/folders/1CBaQWyIV5sb2xLmvvwKS_QKQBOZ7ZEov?usp=sharing

Data
----
Set the dataset path in your config file (see `dataset.dataset_dir` in
`config/default_pathfinder.yaml`). 

The method for downloading the dataset will be sorted out later.

Quick Start
-----------

Train PathFinder:

```bash
python main.py --config default_pathfinder.yaml --mode train
```

Test / Inference:

```bash
python main.py --config default_pathfinder.yaml --mode test
```

Baselines
---------
Baseline configurations are provided under `config/`. To run a baseline, switch
the config file. Example:

```bash
python main.py --config default_radiounet.yaml --mode train
```

The `baseline_main.py` entry point uses a different trainer implementation and
can be used for legacy runs.

Configuration
-------------
Key options in YAML:
- `model.name`: model type (e.g., PathFinder, UNet, RadioUNet, PMNet, REM_Net)
- `opti`: optimizer settings and training schedule
- `dataset`: dataset path and split indices
- `load_pretrain` and `ckp_name`: control checkpoint loading

Outputs
-------
- Checkpoints: `model_path` in config (default `models/`)
- Logs: `log_path` in config (default `logs_pf/` or `logs_rem/`)
- Results: `save_path` in config (default `results/`)

Citation
--------
If you use this code, please cite the paper:

```
@article{zhong2025pathfinder,
title = {PathFinder: Advancing path loss prediction for single-to-multi-transmitter scenario},
journal = {Pattern Recognition},
pages = {113725},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2026.113725},
url = {https://www.sciencedirect.com/science/article/pii/S0031320326006904},
author = {Zhijie Zhong and Zhiwen Yu and Pengyu Li and Jianming Lv and C.L. Philip Chen and Min Chen},
}
```

or

```
@article{zhong2025pathfinder,
  title={PathFinder: Advancing Path Loss Prediction for Single-to-Multi-Transmitter Scenario},
  author={Zhong, Zhijie and Yu, Zhiwen and Li, Pengyu and Lv, Jianming and Chen, CL and Chen, Min},
  journal={arXiv preprint arXiv:2512.14150},
  year={2025}
}
```

License
-------
See the LICENSE file for details.

