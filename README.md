# Visual Language Place Recognition

## Download Dataset
- KITTI dataset </br>
  - image_2 (.png) and ground truth poses (.txt) are required.
</br>
  - download link: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

## Folder Structure
```
${ROOT}
└── lseg/
     └── data/
          └── text_embedding.npy
     └── sripts/
         └── checkpoints/
              └── demo_e200.ckpt
```

## Evaluation
### NetVLAD
#### pittsburgh Dataset
```bash
python main.py --mode=test --resume=<path to checkpoint> --dataset=pittsburgh
```

#### KITTI Dataset
- Used image_2 for the test.

```bash
python main.py --mode=kitti --resume=<path to checkpoint> --dataset=kitti
```

### Our Method
#### Creat Text Embedding
- Input custom label set to create text embedding.
```bash
cd <path to repository>
python build_text_embedding.py
```

#### KITTI Dataset
- `dataset`: Dataset to use. (default: `pittsburgh`, options: `pittsburgh`, `kitti`)
- `build_codebook`: If `True`, generate codebook for BoW. If `False` calculate recall for query images. (default: `False`)

```bash
cd <path to repository>
python extract_pixel_level_embedding.py --dataset=kitti
```
