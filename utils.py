import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

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