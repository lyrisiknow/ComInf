"""Two-stage latent-seed composite-likelihood network inference.

Matrices use [source, target] indexing. Cascades are [cascade, node].
The original MI/community pre-pruning is performed by the caller; its binary
candidate mask is passed as prune_network. No propagation history is assumed.

Usage: from method import ComInf
The default return matches the original (continuous_metrics, seconds, f1).
Set return_details=True for a fourth dictionary containing A, W, rho, seeds,
adjusted frequencies, IPW weights, inferred adjacency and training history.
"""

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class LatentSeedInferenceIC(nn.Module):
    def __init__(self, N, Cascades, InstancePartition, gamma, l1_lambda,
                 prune_network, seed_prior=None, eps=1e-8,
                 w_min=0.1, w_max=10.0):
        super().__init__()
        if not isinstance(N, (int, np.integer)) or N < 2:
            raise ValueError("N must be at least 2 (the loss divides by N - 1).")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        if not (np.isfinite(w_min) and np.isfinite(w_max)
                and 0 < w_min <= w_max):
            raise ValueError("IPW bounds must satisfy 0 < w_min <= w_max.")
        if not (np.isfinite(gamma) and np.isfinite(l1_lambda)
                and gamma >= 0 and l1_lambda >= 0):
            raise ValueError("Regularization coefficients must be nonnegative.")
        self.N, self.gamma, self.l1_lambda = N, gamma, l1_lambda
        self.eps, self.w_min, self.w_max = eps, w_min, w_max
        X = np.asarray(Cascades, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != N or X.shape[0] == 0:
            raise ValueError("Cascades must have nonempty shape (beta, N).")
        if not np.isin(X, [0, 1]).all():
            raise ValueError("Cascades must contain only binary states.")
        if np.any(X.sum(axis=1) == 0):
            raise ValueError("Every cascade must contain an active seed candidate.")
        P = np.asarray(prune_network)
        if P.shape != (N, N) or not np.isfinite(P).all() or np.any(P < 0):
            raise ValueError("prune_network must be a finite nonnegative (N, N) mask.")
        P = (P > 0).astype(np.float32)
        np.fill_diagonal(P, 0)
        self.register_buffer('X', torch.from_numpy(X.copy()))
        self.register_buffer('prune_network_tensor', torch.from_numpy(P))

        # Eq. (3): initialize softplus(Theta) using observed co-activation.
        co = X.T @ X / X.shape[0]
        initial_W = np.clip(0.5 * co, 1e-4, 0.99)
        self.A_param = nn.Parameter(torch.from_numpy(np.log(np.expm1(initial_W))))

        labels = [InstancePartition[u] for u in range(N)]
        # None means an unavailable label, not an additional community.
        communities = list(dict.fromkeys(label for label in labels if label is not None))
        index = {label: k for k, label in enumerate(communities)}
        M = torch.zeros(N, len(communities))
        for u, label in enumerate(labels):
            if label is not None:
                M[u, index[label]] = 1
        self.register_buffer('M_matrix', M)
        self.register_buffer('M_counts', M.sum(0).unsqueeze(1))
        self.register_buffer('known_community_mask', M.sum(1).unsqueeze(1))

        prior = np.ones(N, dtype=np.float64) if seed_prior is None else np.asarray(seed_prior, dtype=np.float64)
        if (prior.shape != (N,) or not np.isfinite(prior).all()
                or np.any(prior < 0) or prior.max() <= 0):
            raise ValueError("seed_prior must be a nonnegative length-N vector with positive mass.")
        prior = prior / prior.max()
        prior = prior / prior.sum()
        # Float64 log priors preserve very small nonzero prior probabilities.
        prior_tensor = torch.from_numpy(prior)
        log_prior = prior_tensor.log()
        active_log_prior = log_prior.unsqueeze(0).expand(X.shape[0], -1).clone()
        active_log_prior.masked_fill_(self.X == 0, -torch.inf)
        normalizer = torch.logsumexp(active_log_prior, dim=1, keepdim=True)
        if not torch.isfinite(normalizer).all():
            raise ValueError("Each cascade needs an active node with positive seed prior.")
        self.register_buffer('seed_prior', prior_tensor)
        self.register_buffer('log_seed_prior', active_log_prior - normalizer)
        self.register_buffer('seed_responsibilities', torch.zeros_like(self.X))
        self.register_buffer('activation_frequency', torch.zeros(N))
        self.register_buffer('ipw_weights', torch.ones(1, N))
        self.register_buffer('activity_penalty_matrix', torch.zeros(N, N))
        self.register_buffer('seeds_frozen', torch.tensor(False))
        self.history = {'stage1': [], 'stage2': []}

    def _get_weights(self):
        """W[j, i] is the intensity from source j to target i, Eq. (3)."""
        return F.softplus(self.A_param) * self.prune_network_tensor

    def _get_prob_matrix(self):
        return -torch.expm1(-self._get_weights())

    def node_losses(self):
        """Eqs. (5), (9); seeds remain present as propagation sources."""
        h = self.X @ self._get_weights()
        return (1 - self.X) * h - self.X * torch.log(-torch.expm1(-h) + self.eps)

    def _seed_logits(self, losses):
        # Eq. (30) has a PLUS local loss; difficult-to-explain active nodes
        # receive larger seed responsibility. Inactive/zero-prior nodes get -inf.
        return self.log_seed_prior + losses

    def marginalized_composite_loss(self):
        """Eq. (19), computed in log space without products/underflow."""
        losses = self.node_losses()
        return (losses.sum(1) - torch.logsumexp(self._seed_logits(losses), dim=1)).mean()

    @torch.no_grad()
    def update_seed_responsibilities(self):
        if self.seeds_frozen.item():
            raise RuntimeError("Stage II seed responsibilities are frozen.")
        rho = torch.softmax(self._seed_logits(self.node_losses()), dim=1)
        self.seed_responsibilities.copy_(rho.to(self.X.dtype))
        return self.seed_responsibilities

    def seed_loss(self):
        """Eq. (32): fixed E-step responsibilities, no IPW or regularization."""
        return ((1 - self.seed_responsibilities) * self.node_losses()).sum() / (self.X.shape[0] * (self.N - 1))

    @torch.no_grad()
    def freeze_seeds_and_compute_ipw(self):
        self.update_seed_responsibilities()
        ordinary = 1 - self.seed_responsibilities
        frequency = (ordinary * self.X).sum(0) / (ordinary.sum(0) + self.eps)
        inverse = 1 / (frequency + self.eps)
        weights = (inverse / inverse.mean()).clamp(self.w_min, self.w_max)
        self.activation_frequency.copy_(frequency)
        self.ipw_weights.copy_(weights.unsqueeze(0))
        # Preserve the original outer-product activity-aware L1 form, using
        # the seed-adjusted frequencies rather than raw activations.
        self.activity_penalty_matrix.copy_(torch.outer(frequency, frequency))
        self.seeds_frozen.fill_(True)

    def structural_regularization(self):
        """Original community regularizer on A, and activity-aware L1 on W."""
        probabilities = self._get_prob_matrix()
        means = (self.M_matrix.T @ probabilities) / self.M_counts
        # Unknown sources neither contribute to community means nor receive a
        # community penalty. Keep the original N^2 scaling and L1 penalty.
        omega = (((probabilities - self.M_matrix @ means) ** 2)
                 * self.known_community_mask).sum() / self.N ** 2
        weighted_l1 = (self._get_weights() * self.activity_penalty_matrix).sum() / self.N ** 2
        return self.gamma * omega + self.l1_lambda * weighted_l1

    def forward(self):
        """Stage II objective, Eqs. (38), (40)."""
        if not self.seeds_frozen.item():
            raise RuntimeError("Complete Stage I and freeze seeds before Stage II.")
        weights = (1 - self.seed_responsibilities) * self.ipw_weights
        fit = (weights * self.node_losses()).sum() / weights.sum()
        return fit + self.structural_regularization()


# Compatibility for code importing the original class name.
CausalInferenceIC = LatentSeedInferenceIC


def _optimize_stage(model, iterations, lr, patience, tol, stage, verbose):
    # Adam without implicit weight decay: all penalties appear in Eq. (40).
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    objective = model.seed_loss if stage == 'stage1' else model.forward
    monitor = model.marginalized_composite_loss if stage == 'stage1' else model.forward
    with torch.no_grad():
        best = float(monitor())
    if not np.isfinite(best):
        raise FloatingPointError(f"Nonfinite initial {stage} objective.")
    best_parameters = model.A_param.detach().clone()
    reference, stale = best, 0
    model.history[stage].append(best)
    for _ in tqdm(range(iterations), desc=stage, disable=not verbose):
        if stage == 'stage1':
            model.update_seed_responsibilities()
        optimizer.zero_grad()
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {stage} loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        # Evaluate AFTER the step so the saved parameters match their score.
        with torch.no_grad():
            value = float(monitor())
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite {stage} objective after optimizer step.")
        model.history[stage].append(value)
        if value < best:
            best = value
            best_parameters = model.A_param.detach().clone()
        if value < reference - tol:
            reference, stale = value, 0
        else:
            stale += 1
        if stale >= patience:
            break
    with torch.no_grad():
        model.A_param.copy_(best_parameters)


def fit_network(N, S, C, gamma, prune_network, iterations=1000, lr=0.01,
                l1_lambda=0.001, *, seed_iterations=500, seed_prior=None,
                w_min=0.1, w_max=10.0, eps=1e-8, patience=100, tol=1e-5,
                device=None, verbose=True):
    """Fit without ground truth; return the trained model and total seconds.

    seed_iterations counts E/M updates (one full-batch Adam step per M update).
    Stage I monitors unweighted MCL. Stage II starts from the restored Stage I
    optimum and never updates its frozen responsibilities or frequency weights.
    """
    if any(not isinstance(v, (int, np.integer)) or v < 1
           for v in (iterations, seed_iterations, patience)):
        raise ValueError("iterations, seed_iterations and patience must be positive integers.")
    if not np.isfinite(lr) or lr <= 0 or not np.isfinite(tol) or tol < 0:
        raise ValueError("lr must be positive and tol must be nonnegative.")
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = LatentSeedInferenceIC(N, S, C, gamma, l1_lambda, prune_network,
                                 seed_prior, eps, w_min, w_max).to(device)
    if verbose:
        print(f"Training on {device}: latent-seed estimation, then seed-aware IPW")
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    model.train()
    _optimize_stage(model, seed_iterations, lr, patience, tol, 'stage1', verbose)
    model.freeze_seeds_and_compute_ipw()
    _optimize_stage(model, iterations, lr, patience, tol, 'stage2', verbose)
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return model, time.perf_counter() - start


def _evaluate(A_star, inferred, G):
    """Evaluate all ordered pairs for directed graphs; upper triangle otherwise."""
    import networkx as nx
    from sklearn.metrics import average_precision_score, roc_auc_score

    N = len(A_star)
    if set(G.nodes()) != set(range(N)):
        raise ValueError("G must contain exactly the node IDs 0, ..., N-1.")
    truth = nx.to_numpy_array(G, nodelist=list(range(N))) > 0
    selection = ~np.eye(N, dtype=bool) if G.is_directed() else np.triu(np.ones((N, N), dtype=bool), k=1)
    y, scores, pred = truth[selection], A_star[selection], inferred[selection]
    continuous = ((round(float(roc_auc_score(y, scores)), 4),
                   round(float(average_precision_score(y, scores)), 4))
                  if np.unique(y).size == 2 else (0.5, 0.0))
    tp = np.count_nonzero(y & pred)
    fp = np.count_nonzero(~y & pred)
    fn = np.count_nonzero(y & ~pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return continuous, tuple(round(v, 3) for v in (precision, recall, f1))


def ComInf(G, N, S, C, A, gamma, prune_network, iterations=1000, lr=0.01,
           l1_lambda=0.001, *, seed_iterations=500, seed_prior=None,
           w_min=0.1, w_max=10.0, eps=1e-8, patience=100, tol=1e-5,
           xi=None, return_details=False, device=None, verbose=True):
    """Compatible entry point; A is retained for compatibility and is unused.

    xi in [0, 1] implements Eq. (42) with a fixed threshold. When xi is None,
    preserve the original community-based thresholding (undirected graphs).
    Directed graphs require xi to avoid the undirected legacy postprocessor.
    G is used only for evaluation, never for network or seed training.
    """
    if xi is not None and (not np.isfinite(xi) or not 0 <= xi <= 1):
        raise ValueError("xi must lie in [0, 1].")
    if xi is None and G.is_directed():
        raise ValueError("Supply xi for directed network inference.")
    model, elapsed = fit_network(
        N, S, C, gamma, prune_network, iterations, lr, l1_lambda,
        seed_iterations=seed_iterations, seed_prior=seed_prior, w_min=w_min,
        w_max=w_max, eps=eps, patience=patience, tol=tol, device=device,
        verbose=verbose)
    with torch.no_grad():
        W = model._get_weights().cpu().numpy()
        A_star = model._get_prob_matrix().cpu().numpy()
        rho = model.seed_responsibilities.cpu().numpy().copy()
        mask = model.prune_network_tensor.cpu().numpy().astype(bool)
    if xi is None:
        import networkx as nx
        from utils import post_processing_with_community
        threshold, graph = post_processing_with_community(A_star, C, beta=1.0)
        inferred = (nx.to_numpy_array(graph, nodelist=list(range(N))) > 0) & mask
    else:
        threshold = xi
        inferred = mask & (A_star >= xi)
    np.fill_diagonal(inferred, False)
    continuous, f1 = _evaluate(A_star, inferred, G)
    result = (continuous, elapsed, f1)
    if verbose:
        print(f"ROC-AUC / PR-AUC: {continuous}; Precision / Recall / F1: {f1}")
    if return_details:
        details = {
            'A': A_star, 'W': W, 'rho': rho,
            'seeds': rho.argmax(axis=1), 'adjacency': inferred.astype(np.int8),
            'activation_frequency': model.activation_frequency.cpu().numpy().copy(),
            'ipw_weights': model.ipw_weights.cpu().numpy().ravel().copy(),
            'threshold': threshold, 'history': model.history,
        }
        result += (details,)
    return result
