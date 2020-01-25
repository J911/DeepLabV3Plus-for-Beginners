# DeepLabV3Plus for Beginners
Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

## Support & Requirements
- pytorch >= 1.1.0
- tensorboardX
- CUDA >= 10.0
- SyncBN support(inplace_abn)

## Install Inplace_abn
```bash
$ pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.10
```

## Train
git clone & change DIR
```bash
$ git clone https://github.com/J911/DeepLabV3Plus-for-Beginners
$ cd DeepLabV3Plus-for-Beginners
```
run 🙌🙌
```bash
$ python -m torch.distributed.launch --nproc_per_node ${num of GPUs} train.py --data /data/CITYSCAPES --batch-size 16 --epoch 200 --logdir ./logs/exp1/ --save ./saved_model/exp1/
```

## Evaluate
```bash
$ python -m torch.distributed.launch --nproc_per_node 1 evaluate.py --data /data/CITYSCAPES --weight ./saved_model/exp1/epoch200.pth --num-classes 19
```

## Dataset

This Repository uses Cityscapes Dataset.

```
CITYSCAPES
|-- leftImg8bit
|   |-- test 
|   |-- train
|   `-- val
`-- gtFine
    |-- test 
    |-- train
    `-- val
```

## Result

TBD

## Thanks to
- @speedinghzl - Gain a lot of Insight 🙇🏻‍♂️
- @mapillary - using inplace_abn