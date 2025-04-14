import networkx as nx
import pickle
import numpy as np

def average_degree(G):
    '''
    Method to compute the average degree of a graph
    
    Params:
    G: Target graph
    
    Returns:
    avg_degree: The average degree of the graph
    '''
    return 2 * G.number_of_edges() / G.number_of_nodes()

def evaluate(G):
    '''
    Method to calculate graph features
    
    Params:
    G: Target graph
    
    Returns:
    transitivity: Graph transitivity
    num_connected: Number of connected components
    avg_deg: Average degree of the graph
    diameter: Graph diameter (inf if unconnected)
    '''
    try:
        diameter = nx.diameter()
    except:
        diameter = np.inf
        
    return nx.transitivity(G), nx.number_connected_components(G), average_degree(G), diameter

if __name__=="__main__":
    file = "fake/10.S.B.M_32"

    with open(f"prob/graph/{file}.pkl", "rb") as f:
        G = pickle.load(f)
        
    print(evaluate(G))
        