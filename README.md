# ComInf

ComInf infers information diffusion networks from final activation snapshots and node community labels, without requiring propagation timestamps or known source nodes. It estimates latent diffusion sources and learns network structure and propagation probabilities through a two-stage inference model.

This repository includes the model implementation, synthetic and real-network experiments, and evaluation tools. Experiments simulate independent cascades on each network, train ComInf, and report network reconstruction performance and source identification accuracy.

## Repository Structure

```text
ComInf/
├── ComInf.py               # ComInf model, training, and evaluation
├── main.py                 # Experiment entry point
├── main_node.py            # LFR network-size experiments
├── utils.py                # Mutual information, pruning, and helper functions
├── requirements.txt        # Python dependencies
├── README.md
└── dataset/
    ├── eu/                 # Email network: edge.txt, department.txt
    ├── workplace/          # Workplace network: edge.txt, department.txt
    └── Mastodon/           # Fakenews24_edges, Fakenews24_instances
```

## Installation

From the `ComInf` directory, install the dependencies:

```bash
pip install -r requirements.txt
pip install tqdm
```

## Datasets

The three real-network datasets are included in `dataset/`. The two LFR synthetic networks are generated at runtime.

| Dataset argument | Network | Nodes |
| --- | --- | ---: |
| `LFR(Small)` | Synthetic, undirected | 100 |
| `LFR(Medium)` | Synthetic, undirected | 1000 |
| `eu` | Email, directed | 1005 |
| `workplace` | Workplace, undirected | 92 |
| `mastodon` | Mastodon, directed | 1340 |


### Dataset Sources

The `eu` and `workplace` datasets are obtained from publicly available external sources:

- **eu (Email-Eu-core):** Obtained from the [Stanford Network Analysis Project (SNAP)](https://snap.stanford.edu/data/email-Eu-core.html). The dataset contains email communication links between members of a European research institution, along with department membership labels.
- **workplace:** Obtained from the [SocioPatterns Project](https://sociopatterns.org/datasets/contacts-in-a-workplace/). The dataset contains face-to-face contact records collected in a workplace, along with department membership labels.

The corresponding network edges and community labels used in our experiments are provided in the `dataset/eu/` and `dataset/workplace/` directories.

## Running Experiments

Run all five datasets with default settings:

```bash
python main.py
```

Run selected datasets with a specified number of cascades:

```bash
python main.py --dataset eu workplace mastodon --n-sim 1000
```

Evaluate different cascade counts:

```bash
python main.py --dataset 'LFR(Small)' --n-sim 50 100 150
```

Each cascade count is applied to every selected dataset. Without `--n-sim`, the number of cascades defaults to the dataset's node count.

Customize training and output settings:

```bash
python main.py --dataset workplace --n-sim 200 \
  --seed-iterations 500 --iterations 10000 \
  --seed 42 --device cpu --output-dir results_method
```

By default, training uses CUDA when available and CPU otherwise. The default maximum update counts are 500 for the first stage and 10,000 for the second stage, with early stopping enabled.

For a quick pipeline check with minimal training:

```bash
python main.py --dataset 'LFR(Small)' --n-sim 12 \
  --seed-iterations 1 --iterations 1 --device cpu --quiet
```

This check verifies execution and result saving; it is not a full training run. View all available options with:

```bash
python main.py --help
```

## Network-Size Experiments

`main.py` evaluates different cascade counts across datasets. `main_node.py`
evaluates different node counts on LFR networks, using the LFR parameters from
`DNIT/mycode/main.py`.

| Default node counts | Default cascades | Average / maximum degree | Community size |
| --- | ---: | --- | --- |
| 100, 150, 200, 250, 300 | 100 | 10 / 30 | 30–60 |
| 1000, 1500, 2000, 2500, 3000 | 1000 | 15 / 50 | 20–50 |

Run the full size sweep or select node counts:

```bash
python main_node.py
python main_node.py --nodes 100 200 300
python main_node.py --nodes 1000 2000 3000
```

Override cascade counts or perform a quick pipeline check:

```bash
python main_node.py --nodes 100 200 300 --n-sim 200
python main_node.py --nodes 100 1000 --seed-iterations 1 --iterations 1 --device cpu --quiet
```

Node counts may be selected within 100–300 or 1000–3000. Training and simulation
settings follow `main.py`; only the LFR generation parameters vary by size range.
Results are saved under `results_node/<UTC timestamp>/LFR_<nodes>/`, using the
same file formats described below. Run `python main_node.py --help` for options.
