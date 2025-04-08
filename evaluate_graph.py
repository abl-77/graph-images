import networkx as nx
import pickle

if __name__=="__main__":
    file = "fake/10.S.B.M"

    with open(f"data/graph/{file}.pkl", "rb") as f:
        G = pickle.load(f)
        
    print(len(G.nodes))
    print(len(G.edges))