# DeepLabV3 for Beginners
Rethinking Atrous Convolution for Semantic Image Segmentation

## Support & Requirements
- 🔥pytorch >= 0.4.0
- 🐍python 3.6.5 
- 📈tensorboardX

- multi GPU support!


### train
git clone & change DIR
```bash
$ git clone https://github.com/J911/DeepLabV3-for-Beginners
$ cd DeepLabV3-for-Beginners
```
run 🙌🙌
```bash
$ python train.py --data /data/CITYSCAPES --batch-size 4 --epoch 1000 --logdir ./logs/exp1/ --save ./saved_model/exp1/
```

### evaluate
```bash
$ python evaluate.py --data /data/CITYSCAPES --weight ./saved_model/epoch80.pth --num-classes 19
```

### Dataset

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

### Result
TBD

## Thanks to
@speedinghzl - Gain a lot of Insight 🙇🏻‍♂️