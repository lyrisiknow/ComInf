"""Run ComInf on all five datasets and save each experiment separately.

Examples:
    python main.py
    python main.py --dataset eu workplace mastodon
    python main.py --dataset 'LFR(Small)' --n-sim 50 100

Default cascade counts: 100 for LFR(Small), 1000 for LFR(Medium),
and the node count for each real network. Real-network cascades are simulated
from the supplied edges and community labels, not observed propagation data.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from ComInf import ComInf
from utils import fast_mi_and_prob, modified_kmeans_fast_log_partitioned


DATASETS = ('LFR(Small)', 'LFR(Medium)', 'eu', 'workplace', 'mastodon')


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', nargs='+', choices=['all', *DATASETS], default=['all'],
                        help='one or more datasets; default: all five')
    parser.add_argument('--datapath', type=Path, default=Path(__file__).resolve().parent / 'dataset')
    parser.add_argument('--n-sim', type=positive_int, nargs='+', help='cascade counts; default: LFR node count or real-network node count')
    parser.add_argument('--seed-iterations', type=positive_int, default=500, help='Stage I: latent-seed estimation updates')
    parser.add_argument('--iterations', type=positive_int, default=10000, help='Stage II: weighted network updates')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--l1-lambda', type=float, default=0.001)
    parser.add_argument('--w-min', type=float, default=0.1)
    parser.add_argument('--w-max', type=float, default=10.0)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--patience', type=positive_int, default=100)
    parser.add_argument('--tol', type=float, default=1e-5)
    parser.add_argument('--xi', type=float, default=0.1, help='fixed edge-probability threshold, Eq. (42)')
    parser.add_argument('--seed-prior', type=Path, help='optional length-N .npy prior in saved node_ids order; default is uniform')
    parser.add_argument('--seed', type=int, default=42, help='random seed for graph generation and simulation')
    parser.add_argument('--device', default=None, help='e.g. cpu or cuda; default selects automatically')
    parser.add_argument('--min-cascade-size', type=positive_int, default=1, help='use 3 for the original simulation filter; 1 retains all single-seed cascades')
    parser.add_argument('--max-attempts', type=positive_int, default=None, help='maximum simulation attempts per cascade count')
    parser.add_argument('--output-dir', type=Path, default=Path('results_method'))
    parser.add_argument('--quiet', action='store_true', help='disable training progress bars')
    return parser


def load_network(dataset, datapath, seed):
    """Return graph and communities with contiguous IDs, plus original IDs."""
    if dataset.startswith('LFR'):
        config = dict(n=100, average_degree=10, max_degree=30, min_community=30, max_community=60)
        if dataset == 'LFR(Medium)':
            config = dict(n=1000, average_degree=15, max_degree=50, min_community=20, max_community=50)
        graph = nx.generators.community.LFR_benchmark_graph(
            **config, tau1=2.0, tau2=2.0, mu=0.1, max_iters=100000, seed=seed)
        original_ids = sorted(graph.nodes())
        labels = [frozenset(graph.nodes[u]['community']) for u in original_ids]
        graph = nx.relabel_nodes(graph, {u: i for i, u in enumerate(original_ids)}, copy=True)
    else:
        files = {
            'eu': ('eu', 'edge.txt', 'department.txt', True, 0),
            'workplace': ('workplace', 'edge.txt', 'department.txt', False, 1),
            'mastodon': ('Mastodon', 'Fakenews24_edges', 'Fakenews24_instances', True, 0),
        }
        if dataset not in files:
            raise ValueError(f'Unknown dataset: {dataset}')
        directory, edge_file, label_file, directed, skip = files[dataset]
        folder = datapath / directory
        labels_by_node = {}
        with (folder / label_file).open(encoding='utf-8-sig') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                fields = line.split(maxsplit=1)
                if len(fields) != 2:
                    raise ValueError(f'{label_file}:{line_number}: expected node and label')
                node, community = fields
                community = community.strip()
                if node in labels_by_node and labels_by_node[node] != community:
                    raise ValueError(f'{label_file}:{line_number}: conflicting labels for {node}')
                labels_by_node[node] = community
        # Label-file order defines reproducible matrix/prior ordering and keeps
        # labeled isolated nodes. String labels (e.g. DISQ) need no conversion.
        original_ids = list(labels_by_node)
        node_index = {node: i for i, node in enumerate(original_ids)}
        graph = nx.DiGraph() if directed else nx.Graph()
        graph.add_nodes_from(range(len(original_ids)))
        with (folder / edge_file).open(encoding='utf-8-sig') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                fields = line.split()
                if len(fields) != skip + 2:
                    raise ValueError(f'{edge_file}:{line_number}: expected {skip + 2} columns')
                source, target = fields[skip:]
                if source not in node_index or target not in node_index:
                    raise ValueError(f'{edge_file}:{line_number}: missing community label for edge {source} -> {target}')
                # Repeated workplace contacts become one undirected edge.
                graph.add_edge(node_index[source], node_index[target])
        labels = [labels_by_node[node] for node in original_ids]
        nx.set_node_attributes(graph, dict(enumerate(labels)), 'original_label')
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    if len(graph) < 2:
        raise ValueError('The network must have at least two nodes.')
    community_index = {label: i for i, label in enumerate(dict.fromkeys(labels))}
    communities = {i: community_index[label] for i, label in enumerate(labels)}
    return graph, communities, np.asarray(original_ids)


def propagation_matrix(graph, communities, rng):
    """Preserve original within/between-community simulation probabilities."""
    A = np.zeros((len(graph), len(graph)), dtype=np.float32)
    for source, target in sorted(graph.edges()):
        bounds = (0.05, 0.1) if communities[source] == communities[target] else (0.1, 0.2)
        A[source, target] = rng.uniform(*bounds)
        if not graph.is_directed():
            A[target, source] = A[source, target]
    return A


def simulate_snapshots(A, num_sim, rng, min_size=1, max_attempts=None):
    """Single-seed independent-cascade simulation, using source/target order.

    Each newly active source attempts each still inactive neighbor once.
    Unlike the legacy utility, this also preserves outgoing directed edges.
    """
    N = len(A)
    if not 1 <= min_size <= N:
        raise ValueError('min-cascade-size must lie between 1 and N.')
    limit = max_attempts if max_attempts is not None else max(10000, num_sim * 1000)
    neighbors = [np.flatnonzero(A[u] > 0) for u in range(N)]
    snapshots = np.zeros((num_sim, N), dtype=np.float32)
    seeds = np.empty(num_sim, dtype=np.int64)
    accepted = 0
    for _ in range(limit):
        seed = int(rng.integers(N))
        active = np.zeros(N, dtype=bool)
        active[seed] = True
        frontier = [seed]
        while frontier:
            next_frontier = []
            for source in frontier:
                targets = neighbors[source]
                targets = targets[~active[targets]]
                successes = targets[rng.random(len(targets)) < A[source, targets]]
                active[successes] = True
                next_frontier.extend(successes.tolist())
            frontier = next_frontier
        if active.sum() < min_size:
            continue
        snapshots[accepted] = active
        seeds[accepted] = seed
        accepted += 1
        if accepted == num_sim:
            return snapshots, seeds
    raise RuntimeError(f'Only generated {accepted}/{num_sim} cascades in {limit} attempts; reduce --min-cascade-size or increase --max-attempts.')


def candidate_mask(snapshots, communities):
    # fast_mi_and_prob returns (conditional probabilities, mutual information).
    _, mi_matrix = fast_mi_and_prob(snapshots.T.astype(np.float64))
    _, fixed_cluster = modified_kmeans_fast_log_partitioned(mi_matrix, communities)
    # With no fixed cluster there is no estimated rejection boundary; retain
    # positive-MI pairs. Zero-MI pairs and self loops are always excluded.
    threshold = max(fixed_cluster.values(), default=0.0)
    mask = (mi_matrix > threshold).astype(np.float32)
    np.fill_diagonal(mask, 0)
    return mask, float(threshold)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (not np.isfinite([args.lr, args.gamma, args.l1_lambda, args.w_min,
                         args.w_max, args.eps, args.tol, args.xi]).all()
            or args.lr <= 0 or args.gamma < 0 or args.l1_lambda < 0
            or not 0 < args.w_min <= args.w_max or args.eps <= 0
            or args.tol < 0 or not 0 <= args.xi <= 1):
        parser.error('Invalid learning rate, regularization, IPW bounds, epsilon, tolerance or threshold.')
    if args.seed < 0:
        parser.error('--seed must be nonnegative.')
    datasets = list(DATASETS) if 'all' in args.dataset else list(dict.fromkeys(args.dataset))
    if args.seed_prior and len(datasets) != 1:
        parser.error('--seed-prior requires exactly one dataset (each has its own node order).')
    # A separate directory per invocation prevents overwriting previous runs.
    batch_dir = args.output_dir / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    batch_dir.mkdir(parents=True, exist_ok=False)
    print(f'Results directory: {batch_dir.resolve()}', flush=True)
    for dataset in datasets:
        try:
            run_dataset(args, dataset, batch_dir)
        except (OSError, ValueError, RuntimeError, nx.ExceededMaxIterations) as exc:
            # Preserve completed experiments and continue with the next dataset.
            error = {'dataset': dataset, 'error': str(exc)}
            with (batch_dir / 'errors.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(error, ensure_ascii=False) + '\n')
            print(f'{dataset} FAILED: {exc}', flush=True)
    if (batch_dir / 'errors.jsonl').exists():
        parser.exit(1, f'Some experiments failed; see {batch_dir / "errors.jsonl"}\n')


def run_dataset(args, dataset, batch_dir, *, network_loader=load_network):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    graph, communities, node_ids = network_loader(dataset, args.datapath, args.seed)
    N = len(graph)
    if args.min_cascade_size > N:
        raise ValueError('--min-cascade-size cannot exceed the node count.')
    A = propagation_matrix(graph, communities, rng)
    seed_prior = np.load(args.seed_prior, allow_pickle=False) if args.seed_prior else None
    if seed_prior is not None and (seed_prior.shape != (N,) or not np.isfinite(seed_prior).all()
                                   or np.any(seed_prior < 0) or not np.any(seed_prior > 0)):
        raise ValueError('--seed-prior must contain a finite nonnegative length-N vector with positive mass.')
    counts = args.n_sim or [N]
    dataset_dir = batch_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f'{dataset}: {N} nodes, {graph.number_of_edges()} edges, {len(set(communities.values()))} communities; directed={graph.is_directed()}')
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings['dataset'] = dataset
    for run, n_sim in enumerate(counts):
        S, true_seeds = simulate_snapshots(A, n_sim, rng, args.min_cascade_size, args.max_attempts)
        prune_network, pruning_threshold = candidate_mask(S, communities)
        print(f'\nCascades={n_sim}; mean active nodes={S.sum(1).mean():.2f}; candidate directed edges={int(prune_network.sum())}')
        auc, seconds, f1, details = ComInf(
            graph, N, S, communities, A, args.gamma, prune_network,
            iterations=args.iterations, lr=args.lr, l1_lambda=args.l1_lambda,
            seed_iterations=args.seed_iterations, seed_prior=seed_prior,
            w_min=args.w_min, w_max=args.w_max, eps=args.eps,
            patience=args.patience, tol=args.tol, xi=args.xi,
            return_details=True, device=args.device, verbose=not args.quiet)
        seed_accuracy = float(np.mean(details['seeds'] == true_seeds))
        # Include count, random seed and run number to distinguish sweep entries.
        name = f'cascades{n_sim}_seed{args.seed}_run{run}'
        artifact = dataset_dir / f'{name}.npz'
        np.savez_compressed(
            artifact, A=details['A'], W=details['W'], rho=details['rho'],
            seeds=details['seeds'], adjacency=details['adjacency'],
            activation_frequency=details['activation_frequency'],
            ipw_weights=details['ipw_weights'], snapshots=S,
            true_seeds=true_seeds, true_probabilities=A, node_ids=node_ids,
            community_labels=np.array([communities[i] for i in range(N)]),
            candidate_mask=prune_network,
            original_labels=np.asarray([str(graph.nodes[i].get('original_label', communities[i])) for i in range(N)]),
            true_adjacency=nx.to_numpy_array(graph, nodelist=list(range(N)), dtype=np.int8))
        record = {
            'method': 'latent-seed-ComInf', 'dataset': dataset, 'n_nodes': N,
            'n_edges': graph.number_of_edges(), 'n_communities': len(set(communities.values())),
            'n_sim': n_sim, 'run': run, 'directed': graph.is_directed(),
            'roc_auc': auc[0], 'pr_auc': auc[1], 'precision': f1[0],
            'recall': f1[1], 'f1': f1[2], 'training_seconds': seconds,
            'seed_accuracy': seed_accuracy, 'pruning_threshold': pruning_threshold,
            'candidate_edges': int(prune_network.sum()), 'artifact': str(artifact.resolve()),
            'settings': settings, 'history': details['history'],
        }
        with (dataset_dir / f'{name}.json').open('w', encoding='utf-8') as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        with (batch_dir / 'result.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f'ROC-AUC={auc[0]:.4f}; PR-AUC={auc[1]:.4f}; F1={f1[2]:.3f}; seed accuracy={seed_accuracy:.3f}; time={seconds:.2f}s')
        print(f'Saved {artifact}')


if __name__ == '__main__':
    main()
