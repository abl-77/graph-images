import networkx as nx
import pickle
import itertools
from collections import Counter
import os
import numpy as np

def get_motifs(graph, k=4):
    '''
    Extract all k-node connected subgraphs (motifs) from the graph
    
    Params:
    graph: networkx Graph object
    k: size of motif (number of nodes)
    
    Returns:
    motif_counts: Counter object of canonical motif identifiers
    '''
    motifs = []
    
    for nodes in itertools.combinations(graph.nodes, k):
        subg = graph.subgraph(nodes)
        if nx.is_connected(subg):
            # Generate a canonical form of the motif (ignores node labels)
            canon_form = nx.weisfeiler_lehman_graph_hash(subg, node_attr=None, edge_attr='weight')
            motifs.append(canon_form)
    
    return Counter(motifs)

def process_directory(graph_dir, k=4):
    '''
    Process all graphs in a directory and compute motif signatures
    
    Params:
    graph_dir: folder with .pkl graph files
    k: motif size
    
    Returns:
    motif_matrix: list of Counter vectors
    file_names: list of graph filenames corresponding to rows
    '''
    motif_matrix = []
    file_names = []

    for root, _, files in os.walk(graph_dir):
        for fname in files:
            if fname.endswith(".pkl"):
                fpath = os.path.join(root, fname)
                with open(fpath, 'rb') as f:
                    G = pickle.load(f)
                
                motifs = get_motifs(G, k)
                motif_matrix.append(motifs)
                file_names.append(fname)
    
    return motif_matrix, file_names

def to_feature_matrix(motif_counters):
    '''
    Convert list of motif Counters into a 2D numpy feature matrix
    '''
    all_keys = sorted(set(key for counter in motif_counters for key in counter.keys()))
    matrix = np.zeros((len(motif_counters), len(all_keys)))

    for i, counter in enumerate(motif_counters):
        for j, key in enumerate(all_keys):
            matrix[i, j] = counter.get(key, 0)
    
    return matrix, all_keys

if __name__=="__main__":
    
    graph_dir = "fake"  # your saved .pkl graphs
    motif_counts, filenames = process_directory(graph_dir, k=1)

    # Convert to feature matrix
    X, motif_ids = to_feature_matrix(motif_counts)

    # You can now analyze X or train a classifier
    print(f"Motif matrix shape: {X.shape}")
    print("First few motif IDs:", motif_ids[:5])
