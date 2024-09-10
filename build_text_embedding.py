import sys
import os
import torch
import clip
import numpy as np

from utils.clip_utils import get_text_feats

path = "/home/data/soojinwoo" # TODO
file_name = "text_embedding.npy" # TODO

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir)

def build_text_embedding_vector(lang):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_version = "ViT-B/32"
    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
    clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
    clip_model.to(device).eval()

    lang = lang.split(",")
    text_feats = get_text_feats(lang, clip_model, clip_feat_dim)

    np.save("{}/{}".format(path, file_name), text_feats)

if __name__=="__main__":
    lang = "ground, road, sidewalk, parking, structure, building, house, vehicle, car, truck, van, bicycle, motorcycle, nature, vegetation, trunk, terrain, tree, human, person, cyclist, bicyclist, motorcyclist, object, fence, pole, traffic, sign, sky, other" # TODO
    build_text_embedding_vector(lang) # TODO input