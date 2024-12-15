import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py

import random
import faiss
import scipy.cluster.vq as vq
from sklearn.cluster import KMeans
import time

root_dir = './data/pittsburgh'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')

struct_dir = join(root_dir, 'datasets/')
queries_dir = join(root_dir, 'queries_real')

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def get_whole_training_set(onlyDB=False):
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform(),
                             onlyDB=onlyDB)

def get_whole_val_set(extract_dataset, random_dataset):
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return WholeDatasetFromStruct(structFile, extract_dataset, random_dataset,
                             input_transform=input_transform())

def get_250k_val_set():
    structFile = join(struct_dir, 'pitts250k_val.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())
def get_whole_test_set():
    structFile = join(struct_dir, 'pitts30k_test.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_250k_test_set():
    structFile = join(struct_dir, 'pitts250k_test.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_training_query_set(margin=0.1):
    structFile = join(struct_dir, 'pitts30k_train.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform(), margin=margin)

def get_val_query_set():
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_250k_val_query_set():
    structFile = join(struct_dir, 'pitts250k_val.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_pitts_dataset_lseg(extract_dataset, random_dataset):
    structFile = join(struct_dir, 'pitts30k_val.mat')
    return PittsDatasetLseg(structFile, extract_dataset, random_dataset,
                             input_transform=input_transform())

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    if '250k' in path.split('/')[-1]:
        dataset = 'pitts250k'
    else:
        dataset = 'pitts30k'

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]
    utmDb = matStruct[2].T

    qImage = [f[0].item() for f in matStruct[3]]
    utmQ = matStruct[4].T

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, 
            utmQ, numDb, numQ, posDistThr, 
            posDistSqThr, nonTrivPosDistSqThr)

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, extract_dataset, random_dataset, input_transform=None, onlyDB=False):
        super().__init__()

        self.extract_dataset = extract_dataset
        self.random_dataset = random_dataset
        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        self.db_dataset = len(self.dbStruct.dbImage)
        self.q_dataset = len(self.dbStruct.qImage)

        if self.extract_dataset: # TODO
            print('===> Extracting partial pittsburgh dataset') # TODO
            num_of_query = 300 # TODO
            self.extracted_db_idx, self.extracted_q_idx = self.extract_partial_dataset(num_of_query)
            self.numDb = self.extracted_db_idx.shape[0]
            self.numQ = num_of_query
        elif self.random_dataset: # TODO
            print('===> Randomizing pittsburgh dataset') # TODO
            random.seed(time.time())
            self.rand_db_idx = random.sample(range(self.db_dataset), self.db_dataset)
            self.rand_q_idx = random.sample(range(self.q_dataset), self.q_dataset)
            self.numDb = len(self.rand_db_idx)
            self.numQ = len(self.rand_q_idx)

        self.db_images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        self.q_images = [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]
       
        if self.extract_dataset:
            self.db_images = [self.db_images[i] for i in self.extracted_db_idx]
            self.q_images = [self.q_images[i] for i in self.extracted_q_idx]
        elif self.random_dataset:
            self.db_images = [self.db_images[i] for i in self.rand_db_idx]
            self.q_images = [self.q_images[i] for i in self.rand_q_idx]
        
        self.images = self.db_images + self.q_images

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

        self.selected_db = None
        self.selected_q = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if self.extract_dataset:
            self.utmDb = [self.dbStruct.utmDb[i] for i in self.extracted_db_idx]
            self.utmQ = [self.dbStruct.utmQ[i] for i in  self.extracted_q_idx]
        elif self.random_dataset:
            self.utmDb = [self.dbStruct.utmDb[i] for i in self.rand_db_idx]
            self.utmQ = [self.dbStruct.utmQ[i] for i in self.rand_q_idx]
        else:
            self.utmDb = self.dbStruct.utmDb
            self.utmQ = self.dbStruct.utmQ

        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.utmDb)

            self.distances, self.positives = knn.radius_neighbors(self.utmQ,
                    radius=self.dbStruct.posDistThr)

        return self.positives
    
    def extract_partial_dataset(self, num_of_query):
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        random.seed(time.time())
        selected_q_idx = random.sample(range(self.q_dataset), num_of_query) # TODO        
        #self.selected_q = self.dbStruct.utmQ[:num_of_query]
        self.selected_q = self.dbStruct.utmQ[selected_q_idx]

        self.distances, self.positives = knn.radius_neighbors(self.selected_q,
                radius=self.dbStruct.posDistThr)

        self.selected_db = self.positives

        selected_db_idx = None
        for db in self.selected_db:
            if selected_db_idx is None:
                selected_db_idx = db
            else:
                selected_db_idx = np.hstack((selected_db_idx, db))
        selected_db_idx = np.unique(selected_db_idx)

        return selected_db_idx, selected_q_idx
    
class PittsDatasetLseg(data.Dataset):
    def __init__(self, structFile, extract_dataset, random_dataset, input_transform=None, onlyDB=False):
        super().__init__()

        self.extract_dataset = extract_dataset
        self.random_dataset = random_dataset
        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        self.db_dataset = len(self.dbStruct.dbImage)
        self.q_dataset = len(self.dbStruct.qImage)

        if self.extract_dataset: # TODO
            print('===> Extracting partial pittsburgh dataset') # TODO
            num_of_query = 300 # TODO
            self.extracted_db_idx, self.extracted_q_idx = self.extract_partial_dataset(num_of_query)
            self.numDb = self.extracted_db_idx.shape[0]
            self.numQ = num_of_query
        elif self.random_dataset: # TODO
            print('===> Randomizing pittsburgh dataset')
            random.seed(time.time())
            self.rand_db_idx = random.sample(range(self.db_dataset), self.db_dataset)
            self.rand_q_idx = random.sample(range(self.q_dataset), self.q_dataset)
            self.numDb = len(self.rand_db_idx)
            self.numQ = len(self.rand_q_idx)

        self.db_images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        self.q_images = [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]
        if self.extract_dataset:
            self.db_images = [self.db_images[i] for i in self.extracted_db_idx]
            self.q_images = [self.q_images[i] for i in self.extracted_q_idx]
        elif self.random_dataset:
            self.db_images = [self.db_images[i] for i in self.rand_db_idx]
            self.q_images = [self.q_images[i] for i in self.rand_q_idx]
        self.images = self.db_images + self.q_images

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
    
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

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if self.extract_dataset:
            self.utmDb = [self.dbStruct.utmDb[i] for i in self.extracted_db_idx]
            self.utmQ = [self.dbStruct.utmQ[i] for i in  self.extracted_q_idx]
        elif self.random_dataset:
            self.utmDb = [self.dbStruct.utmDb[i] for i in self.rand_db_idx]
            self.utmQ = [self.dbStruct.utmQ[i] for i in self.rand_q_idx]
        else:
            self.utmDb = self.dbStruct.utmDb
            self.utmQ = self.dbStruct.utmQ

        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.utmDb)

            self.distances, self.positives = knn.radius_neighbors(self.utmQ,
                    radius=self.dbStruct.posDistThr)

        return self.positives
    
    def extract_partial_dataset(self, num_of_query):
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        random.seed(time.time())
        selected_q_idx = random.sample(range(self.q_dataset), num_of_query) # TODO        
        #self.selected_q = self.dbStruct.utmQ[:num_of_query]
        self.selected_q = self.dbStruct.utmQ[selected_q_idx]

        self.distances, self.positives = knn.radius_neighbors(self.selected_q,
                radius=self.dbStruct.posDistThr)

        self.selected_db = self.positives

        selected_db_idx = None
        for db in self.selected_db:
            if selected_db_idx is None:
                selected_db_idx = db
            else:
                selected_db_idx = np.hstack((selected_db_idx, db))
        selected_db_idx = np.unique(selected_db_idx)

        return selected_db_idx, selected_q_idx
    
    def calculate_recall(self, dbFeat, encoder_dim=10):
        if self.extract_dataset:
            qFeat = dbFeat[self.numDb:].astype('float32') # TODO
            dbFeat = dbFeat[:self.numDb].astype('float32')
        else:
            qFeat = dbFeat[self.dbStruct.numDb:].astype('float32') # TODO
            dbFeat = dbFeat[:self.dbStruct.numDb].astype('float32')
        
        print('====> Building faiss index')
        faiss_index = faiss.IndexFlatL2(encoder_dim)
        faiss_index.add(dbFeat)

        print('====> Calculating recall @ N')
        n_values = [1,5,10,20]

        _, predictions = faiss_index.search(qFeat, max(n_values)) 
        gt = self.getPositives() 

        # for each query get those within threshold distance
        correct_at_n = np.zeros(len(n_values))
        for qIx, pred in enumerate(predictions):
            for i,n in enumerate(n_values):
                if np.any(np.in1d(pred[:n], gt[qIx])):
                    correct_at_n[i:] += 1
                    break
                
        if self.extract_dataset:
            recall_at_n = correct_at_n / self.numQ
        else:
            recall_at_n = correct_at_n / self.dbStruct.numQ

        recalls = {} #make dict for output
        for i,n in enumerate(n_values):
            recalls[n] = recall_at_n[i]
            print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        
def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    
    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices

class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        super().__init__()

        self.input_transform = input_transform
        self.margin = margin

        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample # number of negatives to randomly sample
        self.nNeg = nNeg # number of negatives used for training

        # potential positives are those within nontrivial threshold range
        #fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        # TODO use sqeuclidean as metric?
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.nonTrivPosDistSqThr**0.5, 
                return_distance=False))
        # radius returns unsorted, sort once now so we dont have to later
        for i,posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # its possible some queries don't have any non trivial potential positives
        # lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]

        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.posDistThr, 
                return_distance=False)

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb),
                pos, assume_unique=True))

        self.cache = None # filepath of HDF5 containing feature vectors for images

        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        index = self.queries[index] # re-map index to match dataset
        with h5py.File(self.cache, mode='r') as h5: 
            h5feat = h5.get("features")

            qOffset = self.dbStruct.numDb 
            qFeat = h5feat[index+qOffset]

            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            knn = NearestNeighbors(n_jobs=-1) # TODO replace with faiss?
            knn.fit(posFeat)
            dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            dPos = dPos.item()
            posIndex = self.nontrivial_positives[index][posNN[0]].item()

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))

            negFeat = h5feat[list(map(int, negSample))]
            knn.fit(negFeat)

            dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1), 
                    self.nNeg*10) # to quote netvlad paper code: 10x is hacky but fine
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # try to find negatives that are within margin, if there aren't any return none
            violatingNeg = dNeg < dPos + self.margin**0.5
     
            if np.sum(violatingNeg) < 1:
                #if none are violating then skip this query
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices

        query = Image.open(join(queries_dir, self.dbStruct.qImage[index]))
        positive = Image.open(join(root_dir, self.dbStruct.dbImage[posIndex]))

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            negative = Image.open(join(root_dir, self.dbStruct.dbImage[negIndex]))
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex]+negIndices.tolist()

    def __len__(self):
        return len(self.queries)
