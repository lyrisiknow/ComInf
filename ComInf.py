import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import calculate_continuous_auc, post_processing_with_community, calculate_F1
from tqdm import tqdm
import copy
import time

class CausalInferenceIC(nn.Module):
    def __init__(self, N, Cascades, InstancePartition, gamma, l1_lambda, prune_network):
        super(CausalInferenceIC, self).__init__()
        
        self.N = N
        self.gamma = gamma
        self.l1_lambda = l1_lambda 
        
        prune_network = prune_network.copy()
        prune_network[prune_network == 0] = 1e-5
        self.register_buffer('prune_network_tensor', torch.from_numpy(prune_network).float())

        X_np = np.array(Cascades)
        self.register_buffer('X', torch.tensor(X_np, dtype=torch.float32))

        co_occurrence = np.dot(X_np.T, X_np) / (X_np.shape[0] + 1e-8)
        scaled_co = np.clip(co_occurrence * 0.5, 1e-4, 0.99)
        init_val = np.log(np.exp(scaled_co) - 1.0)
        self.A_param = nn.Parameter(torch.from_numpy(init_val).float())

        unique_insts = list(set(InstancePartition.values()) if isinstance(InstancePartition, dict) else set(InstancePartition))
        num_insts = len(unique_insts)
        M = torch.zeros(N, num_insts, dtype=torch.float32)
        
        for u in range(N):
            inst_id = InstancePartition[u] if isinstance(InstancePartition, dict) else InstancePartition[u]
            idx = unique_insts.index(inst_id)
            M[u, idx] = 1.0
            
        self.register_buffer('M_matrix', M)
        counts = M.sum(dim=0).unsqueeze(1)
        counts[counts == 0] = 1.0 
        self.register_buffer('M_counts', counts)

        node_freq = X_np.mean(axis=0)
        propensity = np.clip(node_freq, 1e-4, 0.95)

        ipw = 1.0 / propensity
        ipw = ipw / ipw.mean()
        self.register_buffer('ipw_weights', torch.tensor(ipw, dtype=torch.float32).unsqueeze(0))
        
        E_matrix = np.outer(propensity, propensity)
        self.register_buffer('causal_penalty_matrix', torch.tensor(E_matrix, dtype=torch.float32))

    def _get_prob_matrix(self):
        W = self._get_weights()
        return -torch.expm1(-W)
    
    def _get_weights(self):
        W = F.softplus(self.A_param)
        
        W = W * (1.0 - torch.eye(self.N, device=W.device))
        W = W * self.prune_network_tensor
        return W

    def forward(self):
        eps = 1e-8
        W = self._get_weights()
        
        Y = torch.matmul(self.X, W.T)
        term1 = (1.0 - self.X) * Y
        log_prob_active = torch.log(-torch.expm1(-Y) + eps)
        term2 = self.X * log_prob_active
        
        node_level_loss = term1 - term2 
        weighted_loss = node_level_loss * self.ipw_weights
        NLL = torch.sum(weighted_loss) / (self.X.shape[0] * self.N)
        
        A_prob = -torch.expm1(-W)
        A_sum = torch.matmul(self.M_matrix.T, A_prob)
        A_mean = A_sum / self.M_counts 
        A_approx = torch.matmul(self.M_matrix, A_mean) 
        Omega = torch.sum((A_prob - A_approx)**2) / (self.N * self.N) 
        
        L1_causal = torch.sum(W * self.causal_penalty_matrix) / (self.N * self.N)
        
        return NLL + (self.gamma * Omega) + (self.l1_lambda * L1_causal)


def ComInf(G, N, S, C, A, gamma, prune_network, iterations=1000, lr=0.01, l1_lambda=0.001):
    patience = 100
    best_loss = float('inf')
    counter = 0 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | Using Causal Full-Batch Gradient Descent")

    model = CausalInferenceIC(N=N, Cascades=S, InstancePartition=C, 
                              gamma=gamma, l1_lambda=l1_lambda, 
                              prune_network=prune_network).to(device)
                                   
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) 
    best_model_wts = copy.deepcopy(model.state_dict())

    model.train()
    pbar = tqdm(range(iterations), desc="Optimizing")
    st = time.time()

    for i in pbar:
        optimizer.zero_grad()

        loss = model() 
        loss.backward() 
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() 
        
        curr_loss = loss.item()
        
        if curr_loss < best_loss - 1e-5:
            best_loss = curr_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            
        if i % 10 == 0:
            pbar.set_postfix({"Loss": f"{curr_loss:.4f}", "Best": f"{best_loss:.4f}", "Patience": f"{counter}/{patience}"})

        if counter >= patience:
            tqdm.write(f"Early stopping at iteration {i+1}. Recovering best weights...")
            break
            
    end_time = time.time() - st
    
    model.load_state_dict(best_model_wts)

    model.eval()
    with torch.no_grad():
        A_star = model._get_prob_matrix().cpu().numpy()

    A_star = A_star * (prune_network > 0).astype(float)
    np.fill_diagonal(A_star, 0.0)

    print("\n--- Evaluation Results ---")
    continuous_metrics = calculate_continuous_auc(A_star, G)
    best_t, IG = post_processing_with_community(A_star, C, beta=1.0) 
    f1 = calculate_F1(IG, G)
    
    return continuous_metrics, end_time, f1