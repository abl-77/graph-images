import networkx as nx
import pickle
import numpy as np
import os
import csv

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
    trans: Graph transitivity
    num_con: Number of connected components
    avg_deg: Average degree of the graph
    diam: Graph diameter or largest subgraph diameter if unconnected
    '''
    largest_cc = max(nx.connected_components(G), key=len).copy()
    sub_g = G.subgraph(largest_cc).copy()
            
    diam = nx.diameter(sub_g)
        
    return nx.transitivity(G), nx.number_connected_components(G), average_degree(G), diam

def evaluate_graphs(folder):
    '''
    Method to evaluate the graph metrics for all graphs in a given folder and write them to a data file
    
    Params:
    folder: path to the image folder
    '''
    # Get the label from the file name
    if "Real" in folder:
        label = "real"
    else:
        label = "synthetic"
        
    data = []
    
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            print(f"Evaluate {filename}")
            
            with open(file_path, "rb") as f:
                G = pickle.load(f) 
            
            trans, num_con, avg_deg, diam = evaluate(G)
            data.append([trans, num_con, avg_deg, diam, label])
    
    with open("data.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

if __name__=="__main__":
    with open("data.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([["transitivity", "number of compontents", "average degree", "diameter", "label"]])
    evaluate_graphs("Probabalistic Graphs/Real faces 64")
    evaluate_graphs("Probabalistic Graphs/Synthetic faces 64")
        