import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from data.dataloader import DataSet
from models.deeplabv3plus import DeepLabV3Plus
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import os
import sys
import argparse

parser = argparse.ArgumentParser(description="DeepLabV3Plus Network")
parser.add_argument("--data", type=str, default="/data/CITYSCAPES", help="")
parser.add_argument("--batch-size", type=int, default=16, help="")
parser.add_argument("--worker", type=int, default=12, help="")
parser.add_argument("--epoch", type=int, default=200, help="")
parser.add_argument("--num-classes", type=int, default=19, help="")
parser.add_argument("--momentum", type=float, default=0.9, help="")
parser.add_argument("--lr", type=float, default=1e-2, help="")
parser.add_argument("--os", type=int, default=16, help="")
parser.add_argument("--weight-decay", type=float, default=5e-4, help="")
parser.add_argument("--logdir", type=str, default="./logs/", help="")
parser.add_argument("--save", type=str, default="./saved_model/", help="")

args = parser.parse_args()

print(args)

writer = SummaryWriter(args.logdir)

train_dataset = DataSet(args.data)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.worker, drop_last=False, shuffle=True, pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = DeepLabV3Plus(num_classes=args.num_classes, os=args.os)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-4)

def train(epoch, iteration, scheduler):
    epoch += 1
    net.train()
    
    train_loss = 0
    for idx, (images, labels) in enumerate(train_loader):
        iteration += 1
        _, h, w = labels.size()

        images, labels = images.to(device), labels.to(device).long()
        out = net(images)
        out = F.interpolate(out, size=(h, w), mode='bilinear')

        loss = criterion(out, labels)

        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("\repoch: ", epoch, "iter: ", (idx + 1), "/", len(train_loader), "loss: ", loss.item(), end='')
        sys.stdout.flush()

    scheduler.step()

    writer.add_scalar('log/loss', train_loss/(idx+1), epoch)
    writer.add_scalar('log/lr', scheduler.get_lr()[0], epoch)

    print("\nepoch: ", epoch, "loss: ", train_loss/(idx+1), "lr: ", scheduler.get_lr()[0])
    
    state = {
        'net': net.module.state_dict(),
        'epoch': epoch,
        'iter': iteration,
    }

    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    saving_path = os.path.join(args.save, 'epoch' + str(epoch) + '.pth')
    torch.save(state, saving_path)

    return epoch, iteration

if __name__=='__main__':
    epoch = 0
    iteration = 0

    while epoch < args.epoch:
        epoch, iteration = train(epoch, iteration, scheduler)
