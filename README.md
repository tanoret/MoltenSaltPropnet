# MoltenSaltPropnet

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python&nbsp;3.10&nbsp;|&nbsp;3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org)
![CI](https://img.shields.io/badge/build-passing-success)

Physics-informed deep-learning toolkit for predicting **thermophysical properties of molten-salt mixtures** from chemical composition and temperature.

---

## Table of Contents
1. [Key Features](#key-features)  
2. [Quick Start](#quick-start)  
3. [Installation](#installation)  
4. [Repository Layout](#repository-layout)  
5. [Data Pipeline](#data-pipeline)  
6. [Training & Inference](#training--inference)  
7. [Contributing](#contributing)  
8. [Roadmap](#roadmap)  
9. [License](#license)  
10. [Citation](#citation)  
11. [Acknowledgements](#acknowledgements)  

---

## Key Features

| Area | What you get |
|------|--------------|
| **Data wrangling** | `MSTDBProcessor` parses, normalizes, and aggregates raw NIST Molten-Salt Thermodynamic Database (MSTDB) CSV tables. |
| **Model zoo** | • `ResNetMetaTrainer` – residual CNN with meta-network correction and physics-based regularization<br>• `SNNMetaTrainer` – sparse NN + meta + physics<br>• `KANTrainer` – kernel attention network baseline |
| **Physics-informed loss** | Automatic enforcement of Arrhenius/polynomial constraints and derivative smoothness. |
| **Turn-key scripts** | Ready-to-run examples in `examples/`: `train_and_predict_resnet.py`, `train_and_predict_snn.py`, `train_and_predict_kan.py`. |
| **Test suite** | PyTest-based unit tests for processors, trainers, and helpers. |
| **Extensible** | Clean, documented API—swap networks, plug in new regularizers, or feed alternative salt databases. |

---

## Quick Start

~~~bash
# 1 · Clone
git clone https://github.com/your-org/MoltenSaltPropnet.git
cd MoltenSaltPropnet

# 2 · Create environment (conda or venv)
conda env create -f environment.yml            # or: python -m venv .venv && source .venv/bin/activate
conda activate mspropnet

# 3 · Install package in editable mode
pip install -e .

# 4 · Train a ResNet + Meta + Physics model
python examples/train_and_predict_resnet.py --epochs 150

# 5 · Predict a property set for a 50-50 NaCl melt
python - << 'EOF'
from processing_mstdb.processor import MSTDBProcessor
from processing_mstdb.resnet_trainer import ResNetMetaTrainer, TARGETS, DERIVED_PROPS

proc = MSTDBProcessor.from_csv("data/mstdb_processed.csv")
trainer = ResNetMetaTrainer(proc.df, TARGETS, DERIVED_PROPS)
trainer.load("outputs/resnet/latest.ckpt")
print(trainer.predict({'Na': 0.5, 'Cl': 0.5, 'T': 973.15}))
EOF
~~~

---

## Installation

### Requirements
* Python 3.10 or 3.11  
* PyTorch ≥ 2.2  
* NumPy, pandas, scikit-learn  
* *(Optional)* CUDA 11+ for GPU acceleration  

~~~bash
pip install -r requirements.txt
~~~

> **Tip:** For GPU use, install the matching **torch** wheel from <https://pytorch.org> *before* running the command above.

---

## Repository Layout

~~~text
MoltenSaltPropnet/
├── data/                     # Raw & processed datasets
│   ├── density-csv.csv
│   ├── viscosity-csv.csv
│   ├── mstdb_processed.csv
│   └── molten-salt-data.pdf
├── processing_mstdb/         # Core library
│   ├── processor.py          # MSTDBProcessor
│   ├── resnet_trainer.py     # ResNetMetaTrainer
│   ├── snn_trainer.py        # SNNMetaTrainer
│   └── kan_trainer.py        # KANTrainer
├── examples/                 # Usage scripts & notebooks
├── tests/                    # PyTest unit tests
├── requirements.txt
├── setup.py / pyproject.toml
└── README.md
~~~

---

## Data Pipeline

1. **Source** – Raw NIST MSTDB CSVs (`density-csv.csv`, `viscosity-csv.csv`, …).  
2. **Pre-processing** – `data_processor.ipynb` cleans columns, harmonizes units, expands polynomial coefficients, and outputs `mstdb_processed.csv`.  
3. **Loading** – `MSTDBProcessor.from_csv(...)` handles:  
   * NaN imputation / type coercion  
   * Composition normalization  
   * Train/validation/test splits with reproducible seeds  
4. **Augmentation** – Optional synthetic smearing in composition–temperature space for better generalization.

Schema and units for every processed column are documented in `docs/DATA_README.md`.

---

## Training & Inference

~~~python
trainer = ResNetMetaTrainer(df, TARGETS, DERIVED_PROPS,
                            lr=3e-3, batch_size=1024)
trainer.train_base()   # stage-1 network
trainer.train_meta()   # stage-2 correction
trainer.save("outputs/resnet/epoch150.ckpt")

y_hat = trainer.predict({"Li": 0.7, "F": 0.3, "T": 993.15})
~~~

* **Physics losses** are activated by default; disable with `physics_weight=0.0`.  
* Logging via **TensorBoard** under `runs/`.  
* See `examples/` for CLI wrappers, hyper-parameter search templates, and notebook demos.

---

## Contributing

Pull requests are welcome! Please:

1. Fork → feature branch → PR.  
2. Follow `black` + `isort` formatting.  
3. Add/update unit tests.  
4. Document public methods with Google-style docstrings.  

For larger changes, open an issue first to discuss the design.

---

## Roadmap

- [ ] **Multi-task uncertainty quantification** (heteroscedastic aleatoric σ-heads)  
- [ ] **Active-learning loop** with experimental design suggestions  
- [ ] **CLI** – `moltsalt-prop train …`, `moltsalt-prop predict …`  
- [ ] **Sphinx docs** auto-generated from docstrings + tutorials  
- [ ] **Web demo** via Streamlit for interactive property lookup  

---

## License

This project is licensed under the **MIT License**—see `LICENSE` for details.

---

## Citation

@software{MoltenSaltPropnet,
author = {Tano, Mauricio and contributors},
title = {MoltenSaltPropnet: Physics-Informed DL for Molten-Salt Properties},
year = {2025},
url = {https://github.com/your-org/MoltenSaltPropnet},
version = {v0.1.0}
}

---

## Acknowledgements

* Raw data derived from **[NIST Janz dataset](https://data.nist.gov/od/id/mds2-2298) and the [Molten-Salt Thermodynamic Database (MSTDB) Thermophysical Properties](https://mstdb.ornl.gov/data-tp/)**.  
* Model inspiration from **ResNet** (He et al., 2016) and **kernel-attention networks**.  

*Happy molten-salt modeling!*
