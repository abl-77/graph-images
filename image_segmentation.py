import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def cluster(A):
    '''
    Method to find a cluster of the most similar nodes in a graph
    
    Params:
    G: weighted similarity graph of an image
    
    Returns:
    Cluster: list of nodes in the cluster
    '''
    # Store the number of nodes for simplicity
    n = A.shape[0]
    # Initialize uniform probability distribution
    x = np.ones(n)/n
    x_prev = np.ones(n)
    
    # Loop until convergence
    while np.sum(np.abs(x_prev - x)) > 0.0001:
        # Store previous probability distribution for convergence check
        x_prev = x
        
        # Calculate weighted averages of node neighbors
        avg = np.matmul(A, x)
        avg = np.asarray(avg).reshape(-1)
        
        # Update probability distributions
        x = (x * avg) / np.dot(x, avg)
        
    # Form cluster of nonzero values
    cluster = np.where(x > 0)[0]
    
    # Clear adjacency matrix for nodes that have already been clustered
    for node in cluster:
        A[node] = np.zeros(n)
        A[:, node] = np.zeros(n)
        
    return cluster
    
if __name__=="__main__":
    file = "fake/10.S.B.M"

    with open(f"data/graph/{file}.pkl", "rb") as f:
        G = pickle.load(f)
        
    # Store the dense adjacency matrix of the graph
    A = np.array(nx.adjacency_matrix(G).todense())
    
    # Create list to store the detected clusters
    clusters = []
    
    # Create parameter to track the number of clustered nodes
    num_clustered = 0
    
    while num_clustered < 0.9 * A.shape[0]:
        clust = cluster(A)
        clusters.append(clust)
        num_clustered += len(clust)
        
    clusters = np.array(clusters, dtype=object)
    
    img = mpimg.imread(f"data/real/{file}.png")
    plt.imshow(img)
    plt.axis("off")  
    plt.show()
    
    img = mpimg.imread(f"data/reduced/{file}.png")
    plt.imshow(img)
    plt.axis("off")  
    plt.show()
