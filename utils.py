import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import json

def calculate_continuous_auc(A_pred, G):
    nodes = sorted(G.nodes())
    adj_true = nx.to_numpy_array(G, nodelist=nodes)
    iu = np.triu_indices(len(nodes), k=1) 
    y_true = (adj_true[iu] > 0).astype(int)
    y_scores = A_pred[iu]
    
    if len(np.unique(y_true)) < 2: 
        return 0.5, 0.0
        
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    print(f"Continuous ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    return round(roc_auc, 4), round(pr_auc, 4)

def post_processing_with_community(estimated_A, node_communities, beta=1.0):
    N = estimated_A.shape[0]
    comm_labels = np.array([node_communities[i] for i in range(N)])
    same_comm_mask = (comm_labels[:, None] == comm_labels[None, :])
    np.fill_diagonal(same_comm_mask, False)
    diff_comm_mask = ~same_comm_mask
    np.fill_diagonal(diff_comm_mask, False)

    mean_inner = np.mean(estimated_A[same_comm_mask])
    mean_outer = np.mean(estimated_A[diff_comm_mask]) + 1e-9
    advantage_ratio = np.clip(mean_inner / mean_outer, 1.0, 5.0) 

    thresholds = np.linspace(1e-6, 0.9, 2000)

    beta_inner = beta * advantage_ratio
    diff_inner = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        pred_fn = np.sum(estimated_A[same_comm_mask & (estimated_A < t)])
        pred_fp = np.sum(1.0 - estimated_A[same_comm_mask & (estimated_A >= t)])
        diff_inner[i] = np.abs(pred_fp - beta_inner * pred_fn)
    t_inner = thresholds[np.argmin(diff_inner)]

    diff_outer = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        pred_fn = np.sum(estimated_A[diff_comm_mask & (estimated_A < t)])
        pred_fp = np.sum(1.0 - estimated_A[diff_comm_mask & (estimated_A >= t)])
        diff_outer[i] = np.abs(pred_fp - beta * pred_fn)
    t_outer = thresholds[np.argmin(diff_outer)]

    print(f"Community Advantage Ratio: {advantage_ratio:.2f}")
    print(f"Optimized Thresholds -> Inner: {t_inner:.5f} | Outer: {t_outer:.5f}")

    IG_mat = np.zeros_like(estimated_A)
    IG_mat[same_comm_mask & (estimated_A >= t_inner)] = 1
    IG_mat[diff_comm_mask & (estimated_A >= t_outer)] = 1
    
    return (t_inner + t_outer) / 2, nx.from_numpy_array(IG_mat)

def calculate_F1(IG, G):
    nodes = sorted(G.nodes())
    adj_predict = nx.to_numpy_array(IG, nodelist=nodes)
    adj_true = nx.to_numpy_array(G, nodelist=nodes)
    
    iu = np.triu_indices(len(nodes), k=1)
    y_true = (adj_true[iu] > 0).astype(int)
    y_pred = adj_predict[iu].astype(int)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    P = TP / (TP + FP + 1e-8)
    R = TP / (TP + FN + 1e-8)
    F1 = 2 * P * R / (P + R + 1e-8)
    
    print(f"Binarized Stats -> TP: {int(TP)}, FP: {int(FP)}, FN: {int(FN)}")
    print(f"Precision: {P:.3f} | Recall: {R:.3f} | F1-Score: {F1:.3f}")
    return round(P, 3), round(R, 3), round(F1, 3)

def result_record(alg_name, ret, dataset, param='', file="result.jsonl"):
    key = f"{dataset}_{alg_name}_{param}" if param else f"{dataset}_{alg_name}"
    
    record_dict = {key: ret}
    
    with open(file, 'a') as f:
        f.write(json.dumps(record_dict) + '\n')
        
def modified_kmeans_fast_log_partitioned(mi_matrix, node_communities, tolerance=1e-7):
    epsilon = 1e-9
    log_mi_matrix = np.log(np.clip(mi_matrix, epsilon, None))

    n = mi_matrix.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    rows_all, cols_all = triu_indices
    all_log_values = log_mi_matrix[triu_indices]
    all_raw_values = mi_matrix[triu_indices]

    same_mask = np.array([
        node_communities.get(r, -1) == node_communities.get(c, -2) 
        for r, c in zip(rows_all, cols_all)
    ])

    def find_active_cluster(log_v, raw_v, r_idx, c_idx):
        if len(log_v) == 0: return {}, {}
        
        fixed_centroid = np.min(log_v) 
        centroid = np.max(log_v)
        
        is_stable = False
        while not is_stable:
            dist_to_fixed = np.abs(log_v - fixed_centroid)
            dist_to_active = np.abs(log_v - centroid)
            active_mask = dist_to_active < dist_to_fixed
            
            new_centroid = np.mean(log_v[active_mask]) if np.any(active_mask) else centroid
            if abs(new_centroid - centroid) < tolerance:
                is_stable = True
            centroid = new_centroid
            
        final_mask = (np.abs(log_v - centroid) < np.abs(log_v - fixed_centroid))
        
        active_dict = dict(zip(zip(r_idx[final_mask], c_idx[final_mask]), raw_v[final_mask]))
        fixed_dict = dict(zip(zip(r_idx[~final_mask], c_idx[~final_mask]), raw_v[~final_mask]))
        return active_dict, fixed_dict

    cluster_same, fixed_same = find_active_cluster(
        all_log_values[same_mask], all_raw_values[same_mask], rows_all[same_mask], cols_all[same_mask]
    )
    cluster_diff, fixed_diff = find_active_cluster(
        all_log_values[~same_mask], all_raw_values[~same_mask], rows_all[~same_mask], cols_all[~same_mask]
    )

    return {**cluster_same, **cluster_diff}, {**fixed_same, **fixed_diff}

def modified_kmeans_fast(mi_matrix, tolerance=1e-7):
    n = mi_matrix.shape[0]

    triu_indices = np.triu_indices(n, k=1)
    all_values = mi_matrix[triu_indices]
    
    valid_mask = all_values > 0
    values = all_values[valid_mask]
    rows = triu_indices[0][valid_mask]
    cols = triu_indices[1][valid_mask]

    fixed_centroid = 0.0
    centroid = np.max(values) if len(values) > 0 else 0.0
    
    is_stable = False
    
    while not is_stable:
        dist_to_fixed = np.abs(values - fixed_centroid)
        dist_to_active = np.abs(values - centroid)
        
        active_mask = dist_to_active < dist_to_fixed
        
        if np.any(active_mask):
            new_centroid = np.mean(values[active_mask])
        else:
            new_centroid = centroid
            
        if abs(new_centroid - centroid) < tolerance:
            is_stable = True
        
        centroid = new_centroid
        
    fixed_mask = ~active_mask
    
    cluster = dict(zip(zip(rows[active_mask], cols[active_mask]), values[active_mask]))
    fixed_cluster = dict(zip(zip(rows[fixed_mask], cols[fixed_mask]), values[fixed_mask]))

    return cluster, fixed_cluster

def fast_mi_and_prob(x):
    n, m = x.shape
    
    count_1 = x.sum(axis=1).get() if hasattr(x, 'get') else x.sum(axis=1)
    count_1 = count_1.astype(float)
    count_0 = m - count_1
    
    p_i1 = count_1 / m
    p_i0 = count_0 / m

    count_11 = x @ x.T
    
    count_1_col = count_1[:, np.newaxis]
    count_1_row = count_1[np.newaxis, :]
    
    count_10 = count_1_col - count_11
    count_01 = count_1_row - count_11
    count_00 = m - (count_11 + count_10 + count_01)

    p_matrix = count_11 / (count_1_col + 1e-12)

    mi_matrix = np.zeros((n, n))
    
    pairs = [
        (count_11, p_i1, p_i1), # (1,1)
        (count_10, p_i1, p_i0), # (1,0)
        (count_01, p_i0, p_i1), # (0,1)
        (count_00, p_i0, p_i0)  # (0,0)
    ]

    for c_ij, p_i_vec, p_j_vec in pairs:
        p_ij = c_ij / m
        p_i_p_j = np.outer(p_i_vec, p_j_vec)
        mask = (p_ij > 1e-12) & (p_i_p_j > 1e-12)

        mi_matrix[mask] += p_ij[mask] * np.log(p_ij[mask] / p_i_p_j[mask])

    return p_matrix, mi_matrix

def Neighbour_finder(g, new_active):
    targets = []
    edges = []
    for node in new_active:
        node_neighbors = list(g.neighbors(node))
        targets += node_neighbors
        for target in node_neighbors:
            edges.append((node,target))

    return (targets, edges)

def IC(Networkx_Graph, Seed_Set, Probability):

    tree = nx.DiGraph()
    tree.add_node(Seed_Set[0])
    new_active, Ans = Seed_Set.tolist(), Seed_Set.tolist()
    while new_active:
        # Getting neighbour nodes of newly activate node
        (targets, edges) = Neighbour_finder(Networkx_Graph, Probability, new_active)
        # Calculating if any nodes of those neighbours can be activated, if yes add them to new_ones.

        new_active = []

        for (node, target) in edges:
            if np.random.uniform(0, 1) < Probability[node, target]:
                if target not in Ans: #success infected
                    tree.add_edge(node, target)
                    new_active.append(target)
                    Ans.append(target)
        # Checking which ones in new_ones are not in our Ans...only adding them to our Ans so that no duplicate in Ans.

    return Ans, tree

def generate_infections(A, num_sim = 100):

    N = A.shape[0]
    S = np.zeros([num_sim, N])
    nx_graph = nx.from_numpy_array(A)
    trees = []
    while len(trees) < num_sim:
        seed = np.random.choice(np.arange(0, N), size=1)
        cascade, tree = IC(Networkx_Graph=nx_graph, Seed_Set=seed, Probability=A)
        if len(tree.nodes) >= 3:
            S[len(trees), cascade] = 1
            trees.append(tree)
    average_paths = 0
    for tree in trees:
        average_paths += len(tree.nodes())

    print("average length of infections: ", average_paths / len(trees))
    return S