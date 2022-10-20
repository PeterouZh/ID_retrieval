# ID_retrieval

ID retrieval is a quantitative criterion to measure the performance of face stylization algorithms. Specifically, it leverages a pre-trained face recognition model to measure the similarity between stylized images and content images. 

**How to calculate ID retrieval:**
*We select the first $100$ images of CelebA-HQ as content images, which are not seen by the face toonify model during training. We randomly synthesize $50$ stylized images for each content image, so there are $5000$ stylized images in total. We use a pre-trained face recognition network to extract face identity vectors for content and stylized images. For each stylized image, we search for its nearest face in the content images and check if the nearest face matches the original content face. The distance adopts the Euclidean distance between the face identity vectors. ID retrieval is the accuracy rate calculated by the proportion of successfully matched images to all stylized images.*

## Quick start

### Dataset structure

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

### Prepare face recognition model

Download face recognition model [model_ir_se50.pth](https://github.com/TreB1eN/InsightFace_Pytorch#2-pretrained-models--performance).
Put `model_ir_se50.pth` in `cache_pretrained/pretrained`.

```text
tree cache_pretrained

cache_pretrained/
└── pretrained
    └── model_ir_se50.pth
```

### Computing ID retrieval

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
ID_retrieval (top1) is the final result. Please refer to `ID_retrieval/scripts/eval.py` for more details. 

## References


| Title                                                                         |         Venue         |  Year |
| :---------------------------------------------------------------------------- | :-------------------: |  :--: |
| [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](http://arxiv.org/abs/1801.07698) | CVPR   | 2019 |
| [FaceShifter: Towards High Fidelity And Occlusion Aware Face Swapping](http://arxiv.org/abs/1912.13457) | CVPR   | 2020 |


## Acknowledgments

- https://github.com/TreB1eN/InsightFace_Pytorch