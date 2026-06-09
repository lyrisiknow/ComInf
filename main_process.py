import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from ComInf import ComInf
import json
from collections import Counter
from utils import result_record, modified_kmeans_fast_log_partitioned, fast_mi_and_prob, IC, generate_infections
import argparse

def get_size_factor(comm_id):
    community_counts = Counter(node_communities.values())
    size = community_counts[comm_id]
    return np.log1p(size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        default="LFR(Small)",
        help="please choose the dataset to use"
    )
    
    args = parser.parse_args()
    dataset = args.dataset
    if "LFR" in dataset:
        if dataset == "LFR(Small)":
            N = 100
            AVG_K = 10  
            MAX_K = 30   
            MIN_C = 30   
            MAX_C = 60
            MU = 0.1    
        
        elif dataset == "LFR(Medium)":
            N = 1000
            AVG_K = 15   
            MAX_K = 50    
            MU = 0.1      
            MIN_C = 20    
            MAX_C = 50    
        else:
            print(f"Error: no dataset -> '{dataset}'")
            return
        
        TAU1 = 2.0    
        TAU2 = 2.0    

        MAX_I = 100000

        # --- Generate LFR Benchmark Graph ---
        try:
            G = nx.generators.community.LFR_benchmark_graph(
                n=N, 
                tau1=TAU1, 
                tau2=TAU2, 
                mu=MU, 
                average_degree=AVG_K, 
                max_degree=MAX_K,       
                min_community=MIN_C, 
                max_community=MAX_C,     
                max_iters=MAX_I,        
                seed=42
            )

            print(f"LFR network generated successfully!")
            print(f"Number of nodes in the generated LFR network: {G.number_of_nodes()}")
            print(f"Number of edges in the generated LFR network: {G.number_of_edges()}")

            # Get ground truth communities
            communities = {frozenset(G.nodes[v]['community']) for v in G}
            print(f"Number of ground truth communities: {len(communities)}")
            
        except nx.ExceededMaxIterations as e:
            print(f"Generation failed: {e}")
            print("Please try adjusting parameters further (e.g., increasing MAX_I or loosening community size constraints slightly).")
    
    unique_comms = sorted(list(set(tuple(G.nodes[n]['community']) for n in G.nodes)))

    comm_to_id = {comm: i for i, comm in enumerate(unique_comms)}

    node_communities = {n: comm_to_id[tuple(G.nodes[n]['community'])] for n in G.nodes}

    A = nx.to_numpy_array(G)
    P = np.zeros((N, N))
    for u, v in G.edges():
        c_u = node_communities[u]
        c_v = node_communities[v]
        if c_u == c_v:
            weight = np.random.uniform(0.05, 0.1)
        else:
            weight = np.random.uniform(0.1, 0.2)
        
        P[u, v] = weight
        P[v, u] = weight
    A = A * P

    if dataset in ['LFR(Small)', 'Workplace']: 
        n_sim_range = [50,100,150,200,250]
    else:
        n_sim_range = [500,1000,1500,2000,2500]
    
    for n_sim in n_sim_range:
        S = generate_infections(A, num_sim=n_sim)
        
        mi_matrix, p_matrix = fast_mi_and_prob(S.T)
        cluster, fixed_cluster = modified_kmeans_fast_log_partitioned(mi_matrix, node_communities)
        threshold = max(fixed_cluster.values())
        prune_network = np.zeros([N, N])
        prune_network[mi_matrix > threshold] = 1.0
        prune_network[mi_matrix <= threshold] = 0.0
        
        G = nx.from_numpy_array(A)
        
        C = node_communities
        l = set()
        for node in C:
            l.add(C[node])
        print(len(l))

        dict_c = dict()
        for i, item in enumerate(l):
            dict_c[item] = i
            
        for node in C:
            C[node] = dict_c[C[node]]
            
        gamma = 0.1
        
        iterations = 10000
        lr = 0.01
        auc, t, f1 = ComInf(G, N, S, C, A, gamma, prune_network, iterations = iterations, lr=lr)
        result_record("ComInf", auc, "LFR", param=f'n{N}auc', file = "result.jsonl")
        result_record("ComInf", f1, "LFR", param=f'n{N}f1', file = "result.jsonl")
        result_record("ComInf", t, "LFR", param=f'n{N}', file = "time.jsonl")    
    

    