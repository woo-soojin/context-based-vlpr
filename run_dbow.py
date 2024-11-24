import os
from os.path import join
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import faiss

root_dir = './'

class DBoW:
    def __init__(self, num_clusters=500):
        self.num_clusters = num_clusters
        self.kmeans = None
        self.vocabulary = None
        self.image_descriptors = list()
        self.image_histograms = list()

        self.images = os.listdir(root_dir)
        # self.images = self.images[:6]
        self.numDb = int(len(self.images)/2) # TODO
        self.numQ = len(self.images) - self.numDb

        # ground truth
        self.gt_pose_path = join(root_dir, 'poses_kitti.txt')
        with open(self.gt_pose_path, 'r') as poses:
            self.utm_coord = [[float(pose.split()[3]), float(pose.split()[7])] for pose in poses]

    def build_vocabulary(self, descriptors_list):
        all_descriptors = np.vstack(descriptors_list)
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)
        self.vocabulary = self.kmeans.cluster_centers_

    def compute_histogram(self, descriptors):
        labels = self.kmeans.predict(descriptors)
        histogram = np.zeros(self.num_clusters)
        for label in labels:
            histogram[label] += 1
        
        return histogram / np.sum(histogram)  # normalization

    def get_positives(self): # TODO
        knn = NearestNeighbors(n_jobs=-1)
        self.utm_coord = np.array(self.utm_coord)
        knn.fit(self.utm_coord[:self.numDb])

        _, positives = knn.radius_neighbors(self.utm_coord[self.numDb:],
                radius=25)

        return positives
    
    def calculate_recall(self, total_histogram):
        qFeat = total_histogram[self.numDb:]
        qFeat = np.array(qFeat)

        dbFeat = total_histogram[:self.numDb]
        dbFeat = np.array(dbFeat)    

        print('====> Building faiss index')
        faiss_index = faiss.IndexFlatL2(self.num_clusters)
        faiss_index.add(dbFeat)

        print('====> Calculating recall @ N')
        n_values = [1,5,10,20]

        _, predictions = faiss_index.search(qFeat, max(n_values))

        # for each query get those within threshold distance
        gt = self.get_positives() 

        correct_at_n = np.zeros(len(n_values))
        for qIx, pred in enumerate(predictions):
            for i,n in enumerate(n_values):
                if np.any(np.in1d(pred[:n], gt[qIx])):
                    correct_at_n[i:] += 1
                    break

        recall_at_n = correct_at_n / self.numQ
        print(recall_at_n)

        recalls = {} #make dict for output # TODO
        for i,n in enumerate(n_values):
            recalls[n] = recall_at_n[i]
            recall_result = "====> Recall@{}: {:.4f}".format(n, recall_at_n[i])
            print(recall_result)

def extract_orb_features(image, nfeatures=500):
    orb = cv2.ORB_create(nfeatures)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

if __name__ == "__main__":
    num_clusters = 10 # TODO
    dbow = DBoW(num_clusters) # num of cluster

    for img_path in dbow.images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, descriptors = extract_orb_features(img) # TODO
        if descriptors is not None:
            print('each', descriptors.shape)
            dbow.image_descriptors.append(descriptors)

    # build vocab
    db_descriptor = dbow.image_descriptors[:dbow.numDb]
    dbow.build_vocabulary(db_descriptor)

    # calculate histogram
    for descriptor in dbow.image_descriptors:
        histogram = dbow.compute_histogram(descriptor)
        dbow.image_histograms.append(histogram)

    dbow.calculate_recall(dbow.image_histograms)