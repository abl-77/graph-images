# Use opencv-python for image processing
import cv2
import networkx as nx
import numpy as np
import pickle

def intensity(pixel):
    '''
    Method to determine the normalized intensity of a pixel
    
    Params:
    pixel: RGB values of a target pixel
    
    Returns:
    intensity: Intensity value of a pixel in the range [0, 1]
    '''
    return np.sum(pixel) / (255 * 3)

def get_neighbors(r, c, width, height):
    '''
    Method to get the neighbor indexes around a central node
    Only count cardinal direction neighbors for now
    
    Params:
    r: Row of the central node
    c: Column of the central node
    width: Width of the image
    height: Height of the image
    
    Returns:
    neighbors: List of neighbor nodes in the form (r, c)
    '''
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            # Check to limit neighbors to cardinal directions
            if np.abs(i + j) == 1:
                if r + i >= 0 and r + i < width and c + j >= 0 and c + j < height:
                    neighbors.append([r + i, c + j])
    
    return neighbors

def get_weight(img, r, c, n_r, n_c, sigma):
    '''
    Method to get the edge weight between two nodes proportional to their intensity similarity
    
    Params:
    img: Intensity matrix of the image
    r: Row of one pixel
    c: Column of one pixel
    n_r: Row of the other pixel
    c_r: Column of the other pixel
    sigma: Decreasing rate of the weights
    
    Returns:
    weight: Weight for a new edge between these two nodes
    '''
    return np.exp(np.abs(img[r, c] - img[n_r, n_c])**2 / sigma)

def get_label(r, c):
    '''
    Method to get a unique node label from pixel coordinates
    
    Params:
    r: Row of the pixel
    c: Column of the pixel
    
    Returns:
    label: Unique node label for graph representation
    '''
    return (1 + r) * (1 + c)
    
def add_edges(graph, img, r, c):
    '''
    Method to add weighted edges to an existing graph based on pixel similarity
    
    Params:
    graph: Undirected weighted graph representation of the image
    img: Intensity matrix of the image
    r: Row of the target node
    c: Column of the target node
    '''
    neighbors = get_neighbors(r, c, img.shape[0], img.shape[1])

    for n_r, n_c in neighbors:
        graph.add_edge(get_label(n_r, n_c), get_label(n_r, n_c), weight=get_weight(img, r, c, n_r, n_c, 1))

def convert_to_graph(path):
    '''
    Method to convert a png image to weighted graph format
    
    Params:
    path: file path to the target image
    
    Returns:
    graph: converted weighted graph of the image
    '''
    # Load image to numpy array
    img = cv2.imread(path)

    # Create intensity image
    img_int = np.zeros(shape=(img.shape[0], img.shape[1]))
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            img_int[r, c] = intensity(img[r, c])
    
    # Create image graph
    img_graph = nx.Graph()
    for r in range(img_int.shape[0]):
        for c in range(img_int.shape[1]):
            add_edges(img_graph, img_int, r, c)
            
    return img_graph
    
if __name__=="__main__":
    file = "fake/10.S.B.M"
    G = convert_to_graph(f"data/{file}.png")
    with open(f"graph/{file}.pkl", "wb") as f:
        pickle.dump(G, f)