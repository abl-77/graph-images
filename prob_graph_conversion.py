import numpy as np
import pickle
import networkx as nx
import cv2
from weight_graph_conversion import intensity, get_label, get_coordinates

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
    img = cv2.imread(f"data/real/{file}.png")
    
    # Reduce image definition
    img = cv2.resize(img, (dim, dim))
    
    # Save reduced definition image
    cv2.imwrite(f"data/reduced/{file}.png", img)    

    # Create intensity image
    img_int = np.zeros(shape=(img.shape[0], img.shape[1]))
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            img_int[r, c] = intensity(img[r, c])
    
    # Create image graph
    img_graph = add_edges(img_int, img.shape[0], img.shape[1])
            
    return img_graph

if __name__=="__main__":
    file = "fake/10.S.B.M"
    dim = 32
    G = convert_to_graph(file, dim)
    with open(f"prob/graph/{file}_{dim}.pkl", "wb") as f:
        pickle.dump(G, f)