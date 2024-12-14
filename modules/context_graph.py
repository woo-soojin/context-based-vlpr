import numpy as np
from scipy import ndimage
import networkx as nx
from scipy.spatial import distance_matrix
from karateclub import Graph2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def calculate_centroids(predicts, dynamic_objects=[]): # TODO default
    unique_values = np.unique(predicts)
    centroids = []

    for value in unique_values:
        if len(dynamic_objects) > 0:
            if str(value) in dynamic_objects: # omit dynamic objects
                continue

        mask = (predicts == value)

        # check connected values
        structure = np.ones((3, 3)) # eight directions
        labeled_array, num_features = ndimage.label(mask, structure=structure)

        # find centroid of cluster
        for i in range(1, num_features + 1):
            centroid = ndimage.center_of_mass(mask, labeled_array, i)
            centroids.append(centroid)

    return centroids

threshold = 70 # TODO
def create_graph(coords, threshold):
    dist_matrix = distance_matrix(coords, coords) # distance matrix

    # adjacent matrix
    # adj_matrix = (dist_matrix <= threshold).astype(int) # binary
    adj_matrix = dist_matrix # weight
    np.fill_diagonal(adj_matrix, 0)  # disconnect self
    G = nx.from_numpy_array(adj_matrix)

    return G, adj_matrix

def calculate_graph_embedding(graphs): # TODO query, db graph
    # model = Graph2Vec(dimensions=64, wl_iterations=2)
    # model = Graph2Vec(dimensions=10, wl_iterations=200) # TODO
    model = Graph2Vec(dimensions=64, wl_iterations=1000) # TODO
    model.fit(graphs)

    embeddings = model.get_embedding() # graph embedding
    return embeddings