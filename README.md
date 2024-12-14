# Context-based Visual Language Place Recognition

## Download Dataset
- KITTI dataset </br>
  - image_2 (.png) and ground truth poses (.txt) are required. </br>
  - [download link](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

## Download Checkpoints
### NetVLAD </br>
- [download link](https://github.com/Nanne/pytorch-NetVlad)
### LSeg </br>
- [download link](https://github.com/isl-org/lang-seg)

## Folder Structure
```
${ROOT}
└── data/
     └── kitti/
          └── 00/
               └── image_2/
               └── poses.txt
     └── pittsburgh/             
└── netvlad/
     └── checkpoints/
└── lseg/
     └── codebook.npy
     └── text_embedding.npy
     └── sripts/
         └── checkpoints/
              └── demo_e200.ckpt
```

## Evaluation
### NetVLAD
#### Pittsburgh Dataset
- `mode`: Select mode. (default: `train`, options: `train`, `test`, `cluster`)
- `resume`: Path to load checkpoint from, for resuming training or testing.
- `dataset`: Dataset to use. (default: `pittsburgh`, options: `pittsburgh`, `kitti`)
- `random`: Randomize dataset for test. (default: `False`)
- `extract_dataset`: Extract partial dataset from whole dataset. (default: `False`)

```bash
python run_netvlad.py --split=val --mode=test --resume=./netvlad --dataset=pittsburgh
```

#### KITTI Dataset
- Use image_2 for the test.
- `mode`: Select mode. (default: `train`, options: `train`, `test`, `cluster`)
- `resume`: Path to load checkpoint from, for resuming training or testing.
- `dataset`: Dataset to use. (default: `pittsburgh`, options: `pittsburgh`, `kitti`)
- `random`: Randomize dataset for test. (default: `False`)

```bash
python run_netvlad.py --split=val --mode=test --resume=./netvlad --dataset=kitti
```

### DBoW
#### KITTI Dataset

```bash
python run_dbow.py
```

### Our Method
#### Creat Text Embedding
- Input custom label set to create text embedding.
```bash
cd <path to repository>
python build_text_embedding.py
```

#### Pittsburgh Dataset
- `data_path`: Path to data. (default: `./data`)
- `dataset`: Dataset to use. (default: `pittsburgh`, options: `pittsburgh`, `kitti`)
- `random`: Randomize dataset for test. (default: `False`)
- `build_codebook`: If `True`, generate codebook for BoW. If `False` calculate recall for query images. (default: `False`)
- `use_codebook`: If `True`, use predefined codebook. (default: `False`)
- `extract_dataset`: Extract partial dataset from whole dataset. (default: `False`)
- `dynamic_objects`: Index of dynamic objects within text embedding
- `save_log`: Save log messages (default: `False`)

```bash
cd <path to repository>
python run_vlpr.py --dataset=pittsburgh
# ex) python run_vlpr.py --dataset=pittsburgh --dynamic_objects 7 8 9 10 11 1 18 19 20 21 22 28
```

#### KITTI Dataset
- `data_path`: Path to data. (default: `./data`)
- `dataset`: Dataset to use. (default: `pittsburgh`, options: `pittsburgh`, `kitti`)
- `random`: Randomize dataset for test. (default: `False`)
- `build_codebook`: If `True`, generate codebook for BoW. If `False` calculate recall for query images. (default: `False`)
- `use_codebook`: If `True`, use predefined codebook. (default: `False`)
- `extract_dataset`: Extract partial dataset from whole dataset. (default: `False`)
- `dynamic_objects`: Index of dynamic objects within text embedding
- `save_log`: Save log messages (default: `False`)

```bash
cd <path to repository>
python run_vlpr.py --dataset=kitti
# ex) python run_vlpr.py --dataset=kitti --dynamic_objects 7 8 9 10 11 12 18 19 20 21 22 28
```

#### Visualize Centroid of Cluster
- Visualization of KITTI 00 Sequence (000001)

<img src="lseg/scripts/images/visualize_centroids_kitti_001.png" alt="centroids_visualization" width="500">


- `image_embedding_file`: Path to image embedding file
- `text_embedding_file`: Path to text embedding file
- `dynamic_objects`: Index of dynamic objects within text embedding

```bash
python visualize_cluster_centroid.py
# ex) python visualize_cluster_centroid.py --dynamic_objects 7 8 9 10 11 12 18 19 20 21 22 28
```
