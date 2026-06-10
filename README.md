# ComInf

This repository provides the official implementation of the **ComInf** model.

---

## Project Overview

**ComInf** is an algorithmic model designed for community-aware information diffusion network inference without timestamps. This repository contains the core source code, evaluation scripts, and real-world datasets used to validate the model's effectiveness.

---

## Core Scripts & Usage

The repository includes two main execution scripts to reproduce the experimental results under different experimental setups:

### 1. Number of Diffusion Process Analysis (`main_process.py`)

* Run this script to evaluate how the performance of the ComInf model scales and changes across **different numbers of propagation processes**.
```bash
python main_process.py

```



### 2. Network Scale Analysis (`main_node.py`)

* Run this script to observe the model's performance and efficiency variations across networks with **different numbers of nodes (network sizes)**.
```bash
python main_node.py

```



---

## Datasets

The paper evaluates the ComInf model across three real-world datasets:

| Dataset | Source / Link | Description |
| --- | --- | --- |
| **Email** | [Dataset Link](https://snap.stanford.edu/data/email-Eu-core.html) | Real-world email communication network. |
| **Workplace** | [Dataset Link](https://sociopatterns.org/datasets.html) | Workplace temporal interaction network. |
| **Mastodon** | Located in the `dataset/` directory | Decentralized social network data (pre-included). |

---

## 🚀 Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/YourUsername/ComInf.git
cd ComInf

```


2. **Install dependencies** (make sure to generate your `requirements.txt` if needed):
```bash
pip install -r requirements.txt

```


3. **Prepare the data**:
* The `mastodon` dataset is already included in the `dataset/` folder.
* For `Email` and `Workplace` datasets, please download them from the links provided above and place them in the appropriate directory.


4. **Run the experiments**:
Execute `main_process.py` or `main_node.py` to view the experimental outcomes.