import numpy as np
import networkx as nx
import pickle as pkl
import os
import cv2
from evaluate_graph import evaluate
import csv

def convert_to_graph(name, prob):
    '''
    Method to convert a png image to weighted graph format
    
    Params:
    name: file name of the target image without .png
    prob: Term for altering the overall frequency of edges
    
    Returns:
    graph: converted weighted graph of the image
    '''
    # Load image and image masks
    if name.split(".")[1] == "S":
        img = cv2.imread(f"Synthetic faces/{name}.png")
        masks = np.load(f"synthetic/{name}_masks.npy")
    else:
        img = cv2.imread(f"Real faces/{name}.png")
        masks = np.load(f"real/{name}_masks.npy")

    # Get average color and position for each segment
    locs = []
    colors = []
    for i in range(masks.shape[0]):
        locs.append(np.average(np.argwhere(masks[i]), axis=0))
        colors.append(np.average(img[masks[i].astype(bool)], axis=0))

    # Create graph
    G = nx.Graph()

    # Check probability of edge for each node combination
    for n in range(masks.shape[0]):
        G.add_node(n)
        for m in range(n + 1, masks.shape[0]):
            color_dif = (765 - np.linalg.norm(colors[n] - colors[m]))
            loc_dif = np.linalg.norm(locs[n] - locs[m])

            if np.random.rand() < prob * (color_dif / loc_dif):
                G.add_edge(n, m)
    
    return G

def convert_folder(folder, prob):
    '''
    Method to convert all images in a folder to graph representations
    
    Params:
    folder: Path to the folder containing the .npy mask files
    prob: Term for altering the overall frequency of edges
    '''
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if ".png" in file_path:
            continue
        if os.path.isfile(file_path):
            base_name = filename[:-10]
            print(f"Convert {base_name}")

            G = convert_to_graph(base_name, prob)
            with open(f"Probabalistic Graphs/{folder}/{filename}.pkl", "wb") as f:
                pkl.dump(G, f)

def compute_metrics(folder, prob, num_graphs):
    '''
    Method to directly compute graph metrics from image and mask files

    Params:
    folder: Path to the folder containing the .npy mask files
    prob: Term for altering the overall frequency of edges
    num_graphs: Number of graphs to generate for averages
    '''
    data = []

    if "synthetic" in folder:
        label = "synthetic"
    else:
        label = "real"

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if ".png" in file_path:
            continue
        if os.path.isfile(file_path):
            base_name = filename[:-10]
            print(f"Convert {base_name}")

            temp = []

            # Create multiple graphs to average the graph metrics
            for i in range(num_graphs):
                G = convert_to_graph(base_name, prob)

                trans, num_con, avg_deg, diam = evaluate(G)
                temp.append([trans, num_con, avg_deg, diam])
            
            temp = np.average(temp, axis=0)
            data.append([temp[0], temp[1], temp[2], temp[3], label])
        
    with open("data.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

if __name__=="__main__":
    with open("data.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([["transitivity", "number of components", "average degree", "diameter", "label"]])
    compute_metrics("synthetic", 0.5, 5)
    # compute_metrics("real", 0.5, 5)
    