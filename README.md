# ID_retrieval

## Quick start

### Prepare data

Please refer to the `datasets` folder.

```text
tree datasets

datasets/
├── content
│   ├── id1.jpg
│   ├── id2.jpg
│   └── xxx.jpg
└── transfer
│   ├── id1
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
│   ├── id2
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
│   ├── xxx
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
└── style (optional, for fid)
    ├── xxx.jpg
    └── xxx.jpg
```

### Computing ID retrieval and FID 

```bash
pip install -r requirements.txt

python ID_retrieval/scripts/eval.py

```
Results:
```text
ID_retrieval (top1)           : 100.00%
ID_retrieval (thresh 1.5)     : 68.75%
FID                           : 229.58

```


## References


| Title                                                                         |         Venue         | Code | Year |
| :---------------------------------------------------------------------------- | :-------------------: | :--: | :--: |
| [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](http://arxiv.org/abs/1801.07698) | CVPR |  | 2019 |
| [FaceShifter: Towards High Fidelity And Occlusion Aware Face Swapping](http://arxiv.org/abs/1912.13457) | CVPR |  | 2020 |


## Acknowledgments

- https://github.com/TreB1eN/InsightFace_Pytorch