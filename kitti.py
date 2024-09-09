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
import random
import time

root_dir = '/home/soojinwoo/' # TODO

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def get_kitti_dataset(random_dataset): # TODO
    image_path = join(root_dir, 'kitti/00/image_2') # TODO
    gt_path = join(root_dir, 'kitti/00') # TODO
    return KittiDatasetNetVLAD(image_path, gt_path, random_dataset,
                             input_transform=input_transform())

class KittiDatasetNetVLAD(data.Dataset):
    def __init__(self, image_path, gt_path, random_dataset, input_transform=None, onlyDB=False): # TODO
        super().__init__()

        self.random_dataset = random_dataset
        self.input_transform = input_transform

        self.whole_image = sorted(os.listdir(image_path))
        self.images = [join(image_path, dbIm) for dbIm in self.whole_image]
        
        # ground truth
        self.gt_pose_path = join(gt_path, 'poses.txt')
        with open(self.gt_pose_path, 'r') as poses:
            self.utm_coord = [[float(pose.split()[3]), float(pose.split()[7])] for pose in poses]
        
        # if not onlyDB: # TODO
        #     self.images += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

        # self.whichSet = self.dbStruct.whichSet
        # self.dataset = self.dbStruct.dataset

        self.positives = None
        # self.distances = None
        self.posDistThr = 25

        self.numDb = int(len(self.images)/2) # TODO
        self.numQ = len(self.images) - self.numDb

        if self.random_dataset:
            print('===> Randomizing kitti dataset')
            random.seed(time.time())
            
            self.total_dataset = len(self.images)
            self.randIdx = random.sample(range(self.total_dataset), self.total_dataset)

            self.images = [self.images[i] for i in self.randIdx]
            self.utm_coord = [self.utm_coord[i] for i in self.randIdx]

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
            self.utm_coord = np.array(self.utm_coord)
            knn.fit(self.utm_coord[:self.numDb])

            self.distances, self.positives = knn.radius_neighbors(self.utm_coord[self.numDb:],
                    radius=self.posDistThr)
        return self.positives
