# Visual Language Place Recognition

## Folder Structure
${ROOT}
└── lseg/    
    └── sripts/
       └── checkpoints/
          └── demo_e200.ckpt

## Evaluation
### NetVLAD
#### KITTI Dataset
```bash
python main.py --mode=kitti --resume=<path to checkpoint> --dataset=kitti
```

### Our Method
#### KITTI Dataset
- `build_codebook`: If `True`, generate codebook for BoW. If `False` calculate recall for query images. (default: `False`)

```bash
cd <path to repository>/lseg/
python main.py
```
