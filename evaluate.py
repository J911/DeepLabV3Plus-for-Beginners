import torch
import torch.nn as nn

from data.dataloader import DataSet
from models.deeplabv3plus import DeepLabV3Plus
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import numpy as np
import os
import sys
import argparse

parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
parser.add_argument("--data", type=str, default="/data/CITYSCAPES", help="")
parser.add_argument("--weight", type=str, default="./saved_model/epoch80.pth", help="")
parser.add_argument("--num-classes", type=int, default=19, help="")
parser.add_argument("--os", type=int, default=16, help="")
parser.add_argument("--local_rank", default=0, type=int, help="")

args = parser.parse_args()

if args.local_rank == 0:
    print(args)

torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl', init_method='env://')

batch_size = 1

test_dataset = DataSet(args.data, train=False, input_size=(1024, 2048), mirror=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=1, drop_last=False, shuffle=False, pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = DeepLabV3Plus(num_classes=args.num_classes, os=args.os)
net = net.to(device)

net = torch.nn.DataParallel(net)
cudnn.benchmark = True
    
checkpoint = torch.load(args.weight)
net.load_state_dict(checkpoint['net'])

def get_confusion_matrix(gt_label, pred_label, class_num):
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def test():
    net.eval()

    confusion_matrix = np.zeros((args.num_classes, args.num_classes))

    for idx, (images, labels) in enumerate(test_loader):
        _, h, w = labels.size()

        images = images.to(device)
        out = net(images)
        out = F.interpolate(out, size=(h, w), mode='bilinear')

        out = torch.argmax(out, dim=1).detach().cpu()

        ignore_index = labels != 255
        out = out[ignore_index]
        labels = labels[ignore_index]

        confusion_matrix += get_confusion_matrix(labels.numpy(), out.numpy(), args.num_classes)

        print("\r[", idx ,"/", len(test_loader) ,"]", end='')
        sys.stdout.flush()

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()

    print("\nmIoU:", mean_IU)
    print(IU_array)

if __name__=='__main__':
    test()
