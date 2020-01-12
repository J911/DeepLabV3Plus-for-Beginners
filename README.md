# DeepLabV3Plus for Beginners
Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

## Support & Requirements
- 🔥pytorch >= 0.4.0
- 🐍python 3.6.5 
- 📈tensorboardX

- multi GPU support!


## Train
git clone & change DIR
```bash
$ git clone https://github.com/J911/DeepLabV3Plus-for-Beginners
$ cd DeepLabV3Plus-for-Beginners
```
run 🙌🙌
```bash
$ python train.py --data /data/CITYSCAPES --batch-size 8 --epoch 200 --logdir ./logs/exp1/ --save ./saved_model/exp1/
```

## Evaluate
```bash
$ python evaluate.py --data /data/CITYSCAPES --weight ./saved_model/epoch80.pth --num-classes 19
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
@speedinghzl - Gain a lot of Insight 🙇🏻‍♂️