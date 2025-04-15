import numpy as np
import pickle
import networkx as nx
import cv2
from weight_graph_conversion import intensity, get_label
import os

# Standardize the random seed
np.random.seed(1)

def get_prob(img, r, c, n_r, n_c):
    '''
    Method to get the probability of an edge between two pixels based on their similarity and distance
    
    Params:
    img: Intensity image representation
    r: Row position of the first pixel
    c: Column position of the first pixel
    n_r: Row position of the second pixel
    n_c: Column position of the second pixel
    
    Reterns:
    p: probability of an edge between the two pixels
    '''
    # Use Euclidian distance
    dist = np.sqrt((r - n_r)**2 + (c - n_c)**2)
    # Similarity is one for the same intensity and less than one for all others
    sim = (255 - np.abs(img[r, c] - img[n_r, n_c])) / 255
    
    return sim/(10 * dist)

def add_edges(img, width, height):
    '''
    Method to add edges to the graph based on pixel similarity and distance
    
    Params:
    img: Intensity matrix of the image
    width: Width of the image
    height: Height of the image
    
    Returns:
    graph: Graph representation of the images
    '''
    graph = nx.Graph()
    for r in range(width):
        for c in range(height):
            # Specifically add all nodes
            graph.add_node(get_label(r, c, width))
            for n_r in range(r + 1, width):
                for n_c in range(c + 1, height):
                    # Calculate p proportional to pixel similarity and inversely proportional to distance
                    p = get_prob(img, r, c, n_r, n_c)
                    
                    if np.random.rand() <= p:
                        graph.add_edge(get_label(r, c, width), get_label(n_r, n_c, width))
                        
    return graph
                    

def convert_to_graph(file, dim):
    '''
    Method to convert a png image to weighted graph format
    
    Params:
    path: file path to the target image
    
    Returns:
    graph: converted weighted graph of the image
    '''
    # Load image to numpy array
    img = cv2.imread(file)
    
    # Reduce image definition
    img = cv2.resize(img, (dim, dim))  

    # Create intensity image
    img_int = np.zeros(shape=(img.shape[0], img.shape[1]))
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            img_int[r, c] = intensity(img[r, c])
    
    # Create image graph
    img_graph = add_edges(img_int, img.shape[0], img.shape[1])
            
    return img_graph

def convert_folder(folder, dim):
    '''
    Method to convert all images in a folder to graph representations
    
    Params:
    folder: Path to the folder containing the images
    dim: Dimension to reduce the images to
    '''
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            print(f"Convert {filename}")
            
            G = convert_to_graph(file_path, dim)
            with open(f"Probabalistic Graphs/{folder} {dim}/{filename}.pkl", "wb") as f:
                pickle.dump(G, f)

if __name__=="__main__":
    dim = 64
    convert_folder("Real faces", dim)
    convert_folder("Synthetic faces", dim)