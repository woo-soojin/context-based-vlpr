import os
from os.path import join, exists

from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
import cv2

# NOTE: you need to download the Nordland dataset from  https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W
# this link is shared and maintained by the authors of VPR_Bench: https://github.com/MubarizZaffar/VPR-Bench
# the folders named ref and query should reside in DATASET_ROOT path
# I hardcoded the image names and ground truth for faster evaluation
# performance is exactly the same as if you use VPR-Bench.

root_dir = '../../data'

class KittiDataset(Dataset):
    def __init__(self, input_transform = None):
        
        image_path = join(root_dir, 'kitti/00/image_2') # TODO
        gt_path = join(root_dir, 'kitti/00') # TODO
        
        self.input_transform = input_transform

        self.whole_image = sorted(os.listdir(image_path))
        self.images = [join(image_path, dbIm) for dbIm in self.whole_image]
        
        # ground truth
        self.gt_pose_path = join(gt_path, 'poses.txt')
        with open(self.gt_pose_path, 'r') as poses:
            self.utm_coord = [[float(pose.split()[3]), float(pose.split()[7])] for pose in poses]
        
        self.positives = None
        self.posDistThr = 25

        self.num_references = int(len(self.images)/2) # TODO
        self.num_queries = len(self.images) - self.num_references

        self.ground_truth = self.get_positives() 
        
    def __getitem__(self, index):
        # img = Image.open(DATASET_ROOT+self.images[index])
        bgr = cv2.imread(self.images[index])
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
    
    def get_positives(self):
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            self.utm_coord = np.array(self.utm_coord)
            knn.fit(self.utm_coord[:self.num_references])

            self.distances, self.positives = knn.radius_neighbors(self.utm_coord[self.num_references:],
                    radius=self.posDistThr)
        return self.positives
