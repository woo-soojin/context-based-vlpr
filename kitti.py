import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import os
from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py

import math

root_dir = '/home/soojinwoo/' # TODO

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def get_kitti_dataset(): # TODO
    image_path = join(root_dir, 'kitti/00/image_2') # TODO
    gt_path = join(root_dir, 'kitti/00') # TODO
    return KittiDataset(image_path, gt_path,
                             input_transform=input_transform())

class KittiDataset(data.Dataset):
    def __init__(self, image_path, gt_path, input_transform=None, onlyDB=False): # TODO
        super().__init__()

        self.input_transform = input_transform

        self.whole_image = sorted(os.listdir(image_path))
        self.images = [join(image_path, dbIm) for dbIm in self.whole_image]
        
        # ground truth
        self.gt_pose_path = join(gt_path, 'poses.txt')
        with open(self.gt_pose_path, 'r') as poses:
            self.utmDb = [[float(pose.split()[3]), float(pose.split()[7])] for pose in poses]
        
        # if not onlyDB: # TODO
        #     self.images += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

        # self.whichSet = self.dbStruct.whichSet
        # self.dataset = self.dbStruct.dataset

        self.positives = None
        # self.distances = None
        self.posDistThr = 25

        self.numDb = int(len(self.images)/2) # TODO
        self.numQ = len(self.images) - self.numDb

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
    
    # def calculate_distance_diff(self, q_pose, gt_pose):
    #     qx = float(q_pose[0])
    #     qy = float(q_pose[1])
    #     qz = float(q_pose[2])

    #     gt_x = float(gt_pose[0])
    #     gt_y = float(gt_pose[1])
    #     gt_z = float(gt_pose[2])

    #     return math.sqrt((gt_x - qx)**2 + (gt_y - qy)**2 + (gt_z - qz)**2)

    # def getPositives(self, idx, thres_distance=1.5): # TODO
    #     positive = list()
    #     q_pose = self.gt_poses[idx]
        
    #     for gt_idx, gt_pose in enumerate(self.gt_poses):
    #         diff = self.calculate_distance_diff(q_pose, gt_pose)

    #         if(diff <= thres_distance):
    #             positive.append(gt_idx)

    #     return np.array(positive)

    def get_positives(self): # TODO
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            self.utmDb = np.array(self.utmDb)
            knn.fit(self.utmDb[:self.numDb])

            self.distances, self.positives = knn.radius_neighbors(self.utmDb[self.numDb:],
                    radius=self.posDistThr)
        return self.positives
