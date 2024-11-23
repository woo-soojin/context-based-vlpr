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

import cv2
import math
import random
import time
import faiss
import scipy.cluster.vq as vq
from sklearn.cluster import KMeans

root_dir = './data' # TODO

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

def get_kitti_dataset_lseg(random_dataset, save_log): # TODO
    image_path = join(root_dir, 'kitti/00/image_2') # TODO
    gt_path = join(root_dir, 'kitti/00') # TODO
    return KittiDatasetLseg(image_path, gt_path, random_dataset, save_log)

class KittiDatasetLseg(data.Dataset):
    def __init__(self, image_path, gt_path, random_dataset, save_log, input_transform=None, onlyDB=False): # TODO
        super().__init__()

        self.random_dataset = random_dataset
        self.save_log = save_log
        self.input_transform = input_transform

        self.whole_image = sorted(os.listdir(image_path))
        self.images = [join(image_path, dbIm) for dbIm in self.whole_image]
        
        # ground truth
        self.gt_pose_path = join(gt_path, 'poses.txt')
        with open(self.gt_pose_path, 'r') as poses:
            self.utm_coord = [[float(pose.split()[3]), float(pose.split()[7]), float(pose.split()[11])] for pose in poses]

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
            self.total_dataset = len(self.images)
            random.seed(time.time())
            self.randIdx = random.sample(range(self.total_dataset), self.total_dataset)

            self.images = [self.images[i] for i in self.randIdx]
            self.utm_coord = [self.utm_coord[i] for i in self.randIdx]

    def __getitem__(self, index):
        bgr = cv2.imread(self.images[index])
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        return img, index

    def __len__(self):
        return len(self.images)
    
    def calculate_distance_diff(self, q_pose, gt_pose):
        qx = float(q_pose[0])
        qy = float(q_pose[1])
        qz = float(q_pose[2])

        gt_x = float(gt_pose[0])
        gt_y = float(gt_pose[1])
        gt_z = float(gt_pose[2])

        return math.sqrt((gt_x - qx)**2 + (gt_y - qy)**2 + (gt_z - qz)**2)

    def get_positives(self): # TODO
        # positive = list()
        # q_pose = self.gt_poses[idx]
        
        # for gt_idx, gt_pose in enumerate(self.gt_poses):
        #     diff = self.calculate_distance_diff(q_pose, gt_pose)

        #     if(diff <= thres_distance):
        #         positive.append(gt_idx)

        # return np.array(positive)

        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            self.utm_coord = np.array(self.utm_coord)
            knn.fit(self.utm_coord[:self.numDb])

            self.distances, self.positives = knn.radius_neighbors(self.utm_coord[self.numDb:],
                    radius=self.posDistThr)
        return self.positives
    
    def build_codebook(self, X, voc_size=10): # TODO voc_size 10
        n, desc, dim = X.shape
        X = X.reshape(n*desc, dim)
        
        kmeans = KMeans(n_clusters= voc_size, n_init=10)
        kmeans.fit(X)
        pred = kmeans.cluster_centers_
        
        return pred
    
    def bag_of_words(self, descriptor, codebook):
        descriptor=descriptor.tolist()
        index = vq.vq(descriptor,codebook)
        hist, _ = np.histogram(index,bins=range(codebook.shape[0] + 1), density=True)
        
        return hist
    
    def bag_of_words_wo_predified_codebook(self, descriptor, num_cluster=100): # TODO # cluster
        kmeans = KMeans(num_cluster, n_init=10)
        kmeans.fit(descriptor)

        codebook = kmeans.predict(descriptor)
        bow_histogram = np.bincount(codebook, minlength=kmeans.n_clusters)

        return bow_histogram

    def calculate_recall(self, dbFeat, encoder_dim=10):
        qFeat = dbFeat[self.numDb:].astype('float32') # TODO
        dbFeat = dbFeat[:self.numDb].astype('float32')
        
        print('====> Building faiss index')
        faiss_index = faiss.IndexFlatL2(encoder_dim)
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
        
        if self.save_log:
            log_messages = ""

        recall_at_n = correct_at_n / self.numQ
        print(recall_at_n)
        if self.save_log:
            log_messages += str(recall_at_n) + '\n'

        recalls = {} #make dict for output # TODO
        for i,n in enumerate(n_values):
            recalls[n] = recall_at_n[i]
            recall_result = "====> Recall@{}: {:.4f}".format(n, recall_at_n[i])
            print(recall_result)
            if self.save_log:
                log_messages += recall_result + '\n'

        if self.save_log:
            log_file = "./data/logs.txt"  # TODO
            with open(log_file, "w") as f:
                f.write(log_messages)

        return recalls

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
            self.utm_coord = [[float(pose.split()[3]), float(pose.split()[7]), float(pose.split()[11])] for pose in poses]
        
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
            self.total_dataset = len(self.images)
            random.seed(time.time())
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
