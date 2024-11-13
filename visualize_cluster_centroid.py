import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse

path = "./data" # TODO

def parse_configs():
    parser = argparse.ArgumentParser(description="config for visualization")

    parser.add_argument('--image_embedding_file', type=str, default='image_embedding.npy', help='image embedding file name')
    parser.add_argument('--text_embedding_file', type=str, default='text_embedding.npy', help='text embedding file name')
    parser.add_argument('--dynamic_objects', nargs='+', default=[], help='index of dynamic objects')
    
    args = parser.parse_args()

    return args

def visualize_centroids(image_embedding, text_embedding, dynamic_objects):
    score = np.einsum('ijk,ai', image_embedding, text_embedding)
    predicts = np.argmax(score, axis=0) # 158, 520

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

    """cmap = ListedColormap([
        [0.7, 0.9, 0.6],
        [0.9, 0.9, 0.6],
        [0.2, 0.8, 1.0],
        [0.9, 0.6, 0.7],
        [0.1, 0.6, 0.8],    
        [0.8, 0.6, 0.9],
        [1.0, 0.8, 0.9],
        [1.0, 0.5, 0.2],
        [0.7, 0.7, 0.8],
        [0.7, 0.7, 1.0],
        [0.9, 0.3, 0.7],
        [0.8, 0.4, 1.0],
    ])""" # TODO for pastel color theme

    # visualization
    num_of_category = text_embedding.shape[0]
    plt.figure(figsize=(10, 10))
    plt.imshow(predicts, vmin=0, vmax=num_of_category-1)
    plt.colorbar(ticks=range(1, num_of_category))

    for centroid in centroids:
        plt.plot(centroid[1], centroid[0], color=(0.09, 1.0, 0.09), marker='o', markersize=5) # visualize center points

    plt.title(f'Clustering Result with Centroids')
    plt.axis('off')
    plt.show()

if __name__=="__main__":
    configs = parse_configs()
    dynamic_objects = configs.dynamic_objects
    image_embedding_file = configs.image_embedding_file
    text_embedding_file = configs.text_embedding_file

    image_embedding = np.load("{}/{}".format(path, image_embedding_file)) # TODO
    text_embedding = np.load("{}/{}".format(path, text_embedding_file)) # TODO

    visualize_centroids(image_embedding, text_embedding, dynamic_objects)