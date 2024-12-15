import sys
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.validation import get_validation_recalls
from dataloaders.boq.PittsburghDataset import PittsburghDataset
from dataloaders.boq.KittiDataset import KittiDataset

def parse_configs():
    parser = argparse.ArgumentParser(description="config for BoQ")
    parser.add_argument('--dataset', type=str, default='pittsburgh', help='Dataset to use', choices=['pittsburgh', 'kitti'])
    parser.add_argument('--batch_size', type=int, default=40, help='batch Size')

    args = parser.parse_args()

    return args

MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]
IM_SIZE = (320, 320)

def input_transform(image_size=IM_SIZE):
    return T.Compose([
        # T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
		T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
        
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

def get_val_dataset(dataset_name, input_transform=input_transform()):
    dataset_name = dataset_name.lower()
    
    if 'pitts' in dataset_name:
        ds = PittsburghDataset(which_ds=dataset_name, input_transform = input_transform)

    elif 'kitti' in dataset_name:
        ds = KittiDataset(input_transform = input_transform)
    
    else:
        raise ValueError
    
    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth

def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        for batch in tqdm(dataloader, 'Calculating descritptors...'):
            imgs, labels = batch
            output, attentions = model(imgs.to(device))
            descriptors.append(output)

    return torch.cat(descriptors)

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq", backbone_name="resnet50", output_dim=16384)
    model = model.to(device)

    val_dataset_name = configs.dataset
    batch_size = configs.batch_size

    val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_dataset_name)
    val_loader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size)

    descriptors = get_descriptors(model, val_loader, device)
    print(f'Descriptor dimension {descriptors.shape[1]}')

    r_list = descriptors[ : num_references].cpu()
    q_list = descriptors[num_references : ].cpu()
    recalls_dict, preds = get_validation_recalls(r_list=r_list,
                                        q_list=q_list,
                                        k_values=[1, 5, 10, 15, 20, 25],
                                        gt=ground_truth,
                                        print_results=True,
                                        dataset_name=val_dataset_name,
                                        )

if __name__ == "__main__":
    configs = parse_configs()

    test()