# Context-based Visual Language Place Recognition

## Download Dataset
- KITTI dataset </br>
  - image_2 (.png) and ground truth poses (.txt) are required. </br>
  - [download link](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

## Download Checkpoints for LSeg
- Pre-trained LSeg model </br>
     - [download link](https://drive.usercontent.google.com/download?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb&authuser=1)

## Folder Structure
```
${ROOT}
└── data/
     └── kitti/
          └── 00/
               └── image_2/
               └── poses.txt
     └── pittsburgh/             
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
python main.py --mode=test --resume=<path to checkpoint> --dataset=pittsburgh
```

#### KITTI Dataset
- Use image_2 for the test.
- `mode`: Select mode. (default: `train`, options: `train`, `test`, `cluster`)
- `resume`: Path to load checkpoint from, for resuming training or testing.
- `dataset`: Dataset to use. (default: `pittsburgh`, options: `pittsburgh`, `kitti`)
- `random`: Randomize dataset for test. (default: `False`)

```bash
python main.py --mode=test --resume=<path to checkpoint> --dataset=kitti
```

### Our Method
#### Creat Text Embedding
- Input custom label set to create text embedding.
```bash
cd <path to repository>
python build_text_embedding.py
```

#### Pittsburgh Dataset
- `dataset`: Dataset to use. (default: `pittsburgh`, options: `pittsburgh`, `kitti`)
- `random`: Randomize dataset for test. (default: `False`)
- `build_codebook`: If `True`, generate codebook for BoW. If `False` calculate recall for query images. (default: `False`)
- `use_codebook`: If `True`, use predefined codebook. (default: `False`)
- `extract_dataset`: Extract partial dataset from whole dataset. (default: `False`)

```bash
cd <path to repository>
python extract_pixel_level_embedding.py --dataset=pittsburgh
```

#### KITTI Dataset
- `dataset`: Dataset to use. (default: `pittsburgh`, options: `pittsburgh`, `kitti`)
- `random`: Randomize dataset for test. (default: `False`)
- `build_codebook`: If `True`, generate codebook for BoW. If `False` calculate recall for query images. (default: `False`)
- `use_codebook`: If `True`, use predefined codebook. (default: `False`)
- `extract_dataset`: Extract partial dataset from whole dataset. (default: `False`)

```bash
cd <path to repository>
python extract_pixel_level_embedding.py --dataset=kitti
```
