"""Evaluate ComInf across LFR network sizes.

LFR parameters follow DNIT/mycode/main.py. Default cascade counts are 100
for 100-300 nodes and 1000 for 1000-3000 nodes.

Examples:
    python main_node.py
    python main_node.py --nodes 100 200 300
    python main_node.py --nodes 1000 2000 3000 --n-sim 1000
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import networkx as nx
import numpy as np

from main import positive_int, run_dataset


DEFAULT_NODES = (100, 150, 200, 250, 300, 1000, 1500, 2000, 2500, 3000)


def lfr_config(n):
    if 100 <= n <= 300:
        sizes = dict(average_degree=10, max_degree=30,
                     min_community=30, max_community=60)
    elif 1000 <= n <= 3000:
        sizes = dict(average_degree=15, max_degree=50,
                     min_community=20, max_community=50)
    else:
        raise ValueError('Node counts must be in 100-300 or 1000-3000.')
    return dict(n=n, tau1=2.0, tau2=2.0, mu=0.1, max_iters=100000, **sizes)


def load_lfr(dataset, datapath, seed):
    # The dataset name keeps results for different sizes in separate folders.
    n = int(dataset.removeprefix('LFR_'))
    graph = nx.generators.community.LFR_benchmark_graph(**lfr_config(n), seed=seed)
    node_ids = sorted(graph.nodes())
    labels = [frozenset(graph.nodes[u]['community']) for u in node_ids]
    index = {label: i for i, label in enumerate(dict.fromkeys(labels))}
    communities = {i: index[label] for i, label in enumerate(labels)}
    graph = nx.relabel_nodes(graph, {u: i for i, u in enumerate(node_ids)}, copy=True)
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    return graph, communities, np.asarray(node_ids)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nodes', type=positive_int, nargs='+', default=list(DEFAULT_NODES),
                        help='node counts in 100-300 or 1000-3000')
    parser.add_argument('--n-sim', type=positive_int, nargs='+',
                        help='override cascade counts for every size; default: 100 or 1000 by size range')
    parser.add_argument('--seed-iterations', type=positive_int, default=500)
    parser.add_argument('--iterations', type=positive_int, default=10000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--l1-lambda', type=float, default=0.001)
    parser.add_argument('--w-min', type=float, default=0.1)
    parser.add_argument('--w-max', type=float, default=10.0)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--patience', type=positive_int, default=100)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--xi', type=float, default=0.1)
    parser.add_argument('--seed-prior', type=Path,
                        help='length-N .npy prior; requires exactly one node count')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None, help='cpu or cuda; default: automatic')
    parser.add_argument('--min-cascade-size', type=positive_int, default=1)
    parser.add_argument('--max-attempts', type=positive_int, default=None)
    parser.add_argument('--output-dir', type=Path, default=Path('results_node'))
    parser.add_argument('--quiet', action='store_true')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.nodes = list(dict.fromkeys(args.nodes))
    try:
        for n in args.nodes:
            lfr_config(n)
    except ValueError as exc:
        parser.error(str(exc))
    if (not np.isfinite([args.lr, args.gamma, args.l1_lambda, args.w_min,
                         args.w_max, args.eps, args.tol, args.xi]).all()
            or args.lr <= 0 or args.gamma < 0 or args.l1_lambda < 0
            or not 0 < args.w_min <= args.w_max or args.eps <= 0
            or args.tol < 0 or not 0 <= args.xi <= 1):
        parser.error('Invalid learning rate, regularization, IPW bounds, epsilon, tolerance or threshold.')
    if args.seed < 0:
        parser.error('--seed must be nonnegative.')
    if args.seed_prior and len(args.nodes) != 1:
        parser.error('--seed-prior requires exactly one node count.')
    if args.min_cascade_size > min(args.nodes):
        parser.error('--min-cascade-size cannot exceed any selected node count.')

    batch_dir = args.output_dir / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    batch_dir.mkdir(parents=True, exist_ok=False)
    print(f'Results directory: {batch_dir.resolve()}', flush=True)
    for n in args.nodes:
        dataset = f'LFR_{n}'
        settings = argparse.Namespace(**vars(args))
        settings.datapath = None
        settings.n_sim = args.n_sim or [100 if n <= 300 else 1000]
        settings.lfr_parameters = dict(lfr_config(n), seed=args.seed)
        try:
            run_dataset(settings, dataset, batch_dir, network_loader=load_lfr)
        except (OSError, ValueError, RuntimeError, nx.ExceededMaxIterations) as exc:
            with (batch_dir / 'errors.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'dataset': dataset, 'n_nodes': n,
                                         'error': str(exc)}) + '\n')
            print(f'{dataset} FAILED: {exc}', flush=True)
    if (batch_dir / 'errors.jsonl').exists():
        parser.exit(1, f'Some experiments failed; see {batch_dir / "errors.jsonl"}\n')


if __name__ == '__main__':
    main()
