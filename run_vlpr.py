import os
import sys
import math

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import clip
import argparse
import yaml
import scipy.cluster.vq as vq
import faiss
from tqdm import tqdm
from scipy import ndimage

from lseg.scripts.additional_utils.models import resize_image, pad_image, crop_image
from lseg.scripts.modules.models.lseg_net import LSegEncNet

from utils.context_graph import calculate_centroids, create_graph, calculate_graph_embedding

import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def parse_configs():
    abs_path = os.path.dirname(os.path.abspath(__file__)) 
    ckpt_path = abs_path + "/lseg/scripts/checkpoints/demo_e200.ckpt"
    parser = argparse.ArgumentParser(description="config for visual-language place recognition")
    parser.add_argument('--data_path', type=str, default=os.path.join(abs_path, 'data'), metavar='PATH', help="the path for dataset required for vlmap creation")
    parser.add_argument('--pretrained_path', type=str, default=ckpt_path, metavar='PATH', help="the path for pretrained checkpoint of lseg")
    parser.add_argument('--mask_version', type=int, default=1, help='mask version | 0 | 1 |, (default: 1)')
    parser.add_argument('--camera_height', type=float, default=1.5, help='height of camera attached to the robot')
    parser.add_argument('--init_tf', nargs='+', action='append', default=[[0.07592686,-0.99711339,0.0,4.872],[0.99711339,0.07592686,0.0,1.24],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]], help='initial transformation of gazebo world')
    parser.add_argument('--rot_ro_cam', nargs='+', action='append', default=[[0, 0, 1],[-1, 0, 0],[0, -1, 0]], help='camera rotation matrix')
    parser.add_argument('--save_pcd', type=bool, default=False, help='the flag to decide whether to save the accumulated point cloud or not')
    parser.add_argument('--dataset', type=str, default='pittsburgh', help='Dataset to use', choices=['pittsburgh', 'kitti'])
    parser.add_argument('--random', type=bool, default=False, help='Randomize dataset for test')
    parser.add_argument('--build_codebook', type=bool, default=False, help='the flag to build codebook')
    parser.add_argument('--use_codebook', type=bool, default=False, help='the flag to use predefined codebook')
    parser.add_argument('--extract_dataset', type=bool, default=False, help='Extract partial dataset from whole dataset') # TODO
    parser.add_argument('--extract_context_graph', type=bool, default=False, help='Extract context graph embedding') # TODO
    parser.add_argument('--use_context_graph', type=bool, default=False, help='Flag to use context graph') # TODO
    parser.add_argument('--dynamic_objects', nargs='+', default=[], help='index of dynamic objects')
    parser.add_argument('--save_log', type=bool, default=False, help='Save log messages')

    args = parser.parse_args()

    return args

def get_lseg_feat(model: LSegEncNet, image: np.array, labels, transform, crop_size=480, \
                 base_size=520, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
    vis_image = image.copy()
    if torch.cuda.is_available():
        image = transform(image).unsqueeze(0).cuda()
    else:
        image = transform(image).unsqueeze(0)
    img = image[0].permute(1,2,0)
    img = img * 0.5 + 0.5

    batch, _, h, w = image.size()
    stride_rate = 2.0/3.0
    stride = int(crop_size * stride_rate)

    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height


    cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})

    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean,
                            norm_std, crop_size)
        with torch.no_grad():
            outputs, logits = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)
    else:
        if short_size < crop_size:
            # pad if needed
            pad_img = pad_image(cur_img, norm_mean,
                                norm_std, crop_size)
        else:
            pad_img = cur_img
        _,_,ph,pw = pad_img.shape #.size()
        assert(ph >= height and pw >= width)
        h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
        with torch.cuda.device_of(image):
            if torch.cuda.is_available():
                with torch.no_grad():
                    outputs = image.new().resize_(batch, model.out_c,ph,pw).zero_().cuda()
                    logits_outputs = image.new().resize_(batch, len(labels),ph,pw).zero_().cuda()
                count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
            else:
                with torch.no_grad():
                    outputs = image.new().resize_(batch, model.out_c,ph,pw).zero_()
                    logits_outputs = image.new().resize_(batch, len(labels),ph,pw).zero_()
                count_norm = image.new().resize_(batch,1,ph,pw).zero_()
        # grid evaluation
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)
                # pad if needed
                pad_crop_img = pad_image(crop_img, norm_mean,
                                            norm_std, crop_size)
                with torch.no_grad():
                    output, logits = model(pad_crop_img, labels)
                cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                cropped_logits = crop_image(logits, 0, h1-h0, 0, w1-w0)
                outputs[:,:,h0:h1,w0:w1] += cropped
                logits_outputs[:,:,h0:h1,w0:w1] += cropped_logits
                count_norm[:,:,h0:h1,w0:w1] += 1
        assert((count_norm==0).sum()==0)
        outputs = outputs / count_norm
        logits_outputs = logits_outputs / count_norm
        outputs = outputs[:,:,:height,:width]
        logits_outputs = logits_outputs[:,:,:height,:width]
    outputs = outputs.cpu()
    outputs = outputs.numpy() # B, D, H, W
    predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
    pred = predicts[0]

    return outputs

def create_lseg_map_batch(pretrained_path, data_dir, camera_height, init_tf, rot_ro_cam, cs=0.05, gs=1000, 
                          crop_size = 480, base_size = 520, lang = "door,chair,ground,ceiling,other",
                          clip_version = "ViT-B/32", mask_version=1):

    labels = lang.split(",")

    # loading models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
    clip_model.to(device).eval()

    model = LSegEncNet(lang, arch_option=0,
                        block_depth=0,
                        activation='lrelu',
                        crop_size=crop_size)
    model_state_dict = model.state_dict()
    
    if torch.cuda.is_available():
        pretrained_state_dict = torch.load(pretrained_path)
    else:
        pretrained_state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))

    pretrained_state_dict = {k.lstrip('net.'): v for k, v in pretrained_state_dict['state_dict'].items()}
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict)

    model.eval()
    model = model.to(device)

    norm_mean= [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    # padding = [0.0] * 3
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # TODO
    if configs.dataset.lower() == 'pittsburgh':
        import dataloaders.pittsburgh as dataset
        whole_test_set = dataset.get_pitts_dataset_lseg(configs.extract_dataset, configs.random) # TODO
        print('Dataset: Pittsburgh')
    elif configs.dataset.lower() == 'kitti':
        import dataloaders.kitti as dataset
        whole_test_set = dataset.get_kitti_dataset_lseg(configs.random, configs.save_log)
        print('Dataset: Kitti')
    else:
        raise Exception('Unknown dataset')

    threads = 8
    cacheBatchSize = 1
    cuda = True
    # TODO what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=whole_test_set,
                num_workers=threads, batch_size=cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

    print('====> Extracting Features')
    print("codebook builder" if configs.build_codebook else "recall calculator")

    encoder_dim = 10 # TODO
    text_embedding_vectors = np.load("{}/{}".format(data_dir, "text_embedding.npy")) # TODO
    if configs.build_codebook: # TODO
        total_descriptors = np.empty((len(whole_test_set), 1000, 512)) # TODO size
    elif configs.use_codebook:
        codebook = np.load("{}/{}".format(data_dir, "codebook.npy"))
        encoder_dim = 10 # TODO
        dbFeat = np.empty((len(whole_test_set), encoder_dim))
    elif not configs.use_codebook:
        encoder_dim = 10 # TODO
        dbFeat = np.empty((len(whole_test_set), encoder_dim))

    if configs.extract_context_graph:
        total_graph = list()
    elif configs.use_context_graph:
        context_graph_embedding_vectors = np.load("{}/{}".format(data_dir, "context_graph_embeddings_64_wo_dynamic.npy")) # TODO

    for iteration, (input, indices) in enumerate(test_data_loader, 1):
        input = input.detach().cpu().numpy()
        input = np.squeeze(input)

        if configs.dataset.lower() == 'pittsburgh': # TODO
            input = input.transpose(1,2,0)

        pix_feats = get_lseg_feat(model, input, labels, transform, crop_size, base_size, norm_mean, norm_std)
        pix_feats = np.squeeze(pix_feats)

        score = np.einsum('ijk,ai',pix_feats, text_embedding_vectors)
        predicts = np.argmax(score, axis=0)
        
        if configs.extract_context_graph:
            threshold = 70 # TODO
            centroids = calculate_centroids(predicts)
            graph, _ = create_graph(centroids, threshold)
            total_graph.append(graph)
        else:
            # mask = (predicts != 10) # others, TODO
            # mask = (predicts != 0) & (predicts != 1) & (predicts != 2) # TODO
            mask = np.ones_like(predicts, dtype=bool)
            if len(configs.dynamic_objects) > 0: # filtering
                for dynamic_object in configs.dynamic_objects:
                    mask &= (predicts != int(dynamic_object))

            filtered_feats = pix_feats[:,mask]
            np.random.shuffle(filtered_feats) # TODO
            fixed_feats = filtered_feats[:, :1000]
            query_descriptor = fixed_feats.T

            if configs.build_codebook:
                total_descriptors[indices,:,:] = query_descriptor
            elif configs.use_codebook:
                query_histogram = whole_test_set.bag_of_words(query_descriptor, codebook)
                if configs.use_context_graph:
                    context_graph_embedding = context_graph_embedding_vectors[indices]
                    merged_embedding = np.hstack((query_histogram, context_graph_embedding))
                    dbFeat[indices, :] = merged_embedding
                else:
                    dbFeat[indices, :] = query_histogram

            elif not configs.use_codebook:
                query_histogram = whole_test_set.bag_of_words_wo_predified_codebook(query_descriptor, 10) # TODO 10
                if configs.use_context_graph:
                    context_graph_embedding = context_graph_embedding_vectors[indices]
                    merged_embedding = np.hstack((query_histogram, context_graph_embedding))
                    dbFeat[indices, :] = merged_embedding
                else:
                    dbFeat[indices, :] = query_histogram
        
    if configs.build_codebook:
        print('====> Building Codebook')
        codebook = whole_test_set.build_codebook(total_descriptors)
        np.save("{}/{}".format(data_dir, "codebook.npy"), codebook) # TODO
    elif configs.extract_context_graph: # TODO
        print('====> Building Context Graph Embeddings')
        context_graph_embeddings = calculate_graph_embedding(total_graph)
        np.save("{}/{}".format(data_dir, "context_graph_embeddings_64_wo_dynamic.npy"), context_graph_embeddings) # TODO
    else:
        whole_test_set.calculate_recall(dbFeat, encoder_dim)
           
if __name__ == "__main__":
    configs = parse_configs()

    data_dir = configs.data_path
    pretrained_path = configs.pretrained_path

    init_tf = np.array(configs.init_tf)
    rot_ro_cam = np.array(configs.rot_ro_cam)
    data_path = configs.data_path
    save_pcd = configs.save_pcd

    camera_height = 1.2 # TODO
    map_resolution = 0.05  # TODO
    create_lseg_map_batch(pretrained_path, data_dir, camera_height, init_tf, rot_ro_cam, cs=map_resolution, gs=1000, mask_version=configs.mask_version)
