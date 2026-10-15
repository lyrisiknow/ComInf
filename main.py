"""Run the two-stage method over different numbers of diffusion snapshots.

Examples:
    python main.py --dataset 'LFR(Small)'
    python main.py --dataset 'LFR(Small)' --n-sim 50 100 --seed-iterations 500 --iterations 10000
    python main.py --dataset mastodon --datapath ./dataset --xi 0.1

Each run writes a metrics record and an NPZ containing probabilities, seed
responsibilities and original node IDs. Simulated seed identities are used
only for evaluation; the method receives no ground-truth seed information.
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from method import ComInf
from utils import fast_mi_and_prob, modified_kmeans_fast_log_partitioned


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=['LFR(Small)', 'LFR(Medium)', 'mastodon'], default='LFR(Small)')
    parser.add_argument('--datapath', type=Path, default=Path(__file__).resolve().parent / 'dataset')
    parser.add_argument('--n-sim', type=positive_int, nargs='+', help='cascade counts; defaults to the original dataset-specific sweep')
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
        folder = datapath / 'Mastodon'
        labels_by_node = {}
        with (folder / 'Fakenews24_instances').open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    node, community = line.split()
                    labels_by_node[node] = community
        # File order defines the reproducible matrix/prior ordering.
        original_ids = list(labels_by_node)
        node_index = {node: i for i, node in enumerate(original_ids)}
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(original_ids)))
        with (folder / 'Fakenews24_edges').open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                source, target = line.split()
                if source not in node_index or target not in node_index:
                    raise ValueError(f'Missing community label for edge {source} -> {target}')
                graph.add_edge(node_index[source], node_index[target])
        labels = [labels_by_node[node] for node in original_ids]
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
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    try:
        graph, communities, node_ids = load_network(args.dataset, args.datapath, args.seed)
    except (OSError, ValueError, nx.ExceededMaxIterations) as exc:
        parser.error(str(exc))
    N = len(graph)
    if args.min_cascade_size > N:
        parser.error('--min-cascade-size cannot exceed the node count.')
    A = propagation_matrix(graph, communities, rng)
    seed_prior = np.load(args.seed_prior, allow_pickle=False) if args.seed_prior else None
    if seed_prior is not None and (seed_prior.shape != (N,) or not np.isfinite(seed_prior).all()
                                   or np.any(seed_prior < 0) or not np.any(seed_prior > 0)):
        parser.error('--seed-prior must contain a finite nonnegative length-N vector with positive mass.')
    counts = args.n_sim or ([50, 100, 150, 200, 250] if args.dataset == 'LFR(Small)' else [500, 1000, 1500, 2000, 2500])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'{args.dataset}: {N} nodes, {graph.number_of_edges()} edges, {len(set(communities.values()))} communities; directed={graph.is_directed()}')
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
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
        name = f'{args.dataset}_cascades{n_sim}_seed{args.seed}_run{run}'
        artifact = args.output_dir / f'{name}.npz'
        np.savez_compressed(
            artifact, A=details['A'], W=details['W'], rho=details['rho'],
            seeds=details['seeds'], adjacency=details['adjacency'],
            activation_frequency=details['activation_frequency'],
            ipw_weights=details['ipw_weights'], snapshots=S,
            true_seeds=true_seeds, true_probabilities=A, node_ids=node_ids,
            community_labels=np.array([communities[i] for i in range(N)]),
            candidate_mask=prune_network)
        record = {
            'method': 'latent-seed-ComInf', 'dataset': args.dataset, 'n_nodes': N,
            'n_sim': n_sim, 'run': run, 'directed': graph.is_directed(),
            'roc_auc': auc[0], 'pr_auc': auc[1], 'precision': f1[0],
            'recall': f1[1], 'f1': f1[2], 'training_seconds': seconds,
            'seed_accuracy': seed_accuracy, 'pruning_threshold': pruning_threshold,
            'candidate_edges': int(prune_network.sum()), 'artifact': str(artifact),
            'settings': settings, 'history': details['history'],
        }
        with (args.output_dir / 'result.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f'ROC-AUC={auc[0]:.4f}; PR-AUC={auc[1]:.4f}; F1={f1[2]:.3f}; seed accuracy={seed_accuracy:.3f}; time={seconds:.2f}s')
        print(f'Saved {artifact}')


if __name__ == '__main__':
    main()
