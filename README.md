# Visual Language Place Recognition

## Folder Structure
```
${ROOT}
└── lseg/
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
#### KITTI Dataset
- `dataset`: Dataset to use. (default: `pittsburgh`, options: `pittsburgh`, `kitti`)
- `build_codebook`: If `True`, generate codebook for BoW. If `False` calculate recall for query images. (default: `False`)

```bash
cd <path to repository>/lseg/
python main.py --dataset=kitti
```
