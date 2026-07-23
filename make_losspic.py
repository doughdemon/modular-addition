import argparse

import torch

from utils.checkpoints import *
from utils.config import *
from utils.figure_mpl import *
from utils.model import *

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(prog='make_losspic.py', description='Show the loss curve')
parser.add_argument('task', help="task")
parser.add_argument('-f', '--file', default='loss.png')
parser.add_argument('-n', '--number', default=100, help="number of epochs")

args = parser.parse_args()

CHECKPOINT=100

_, _, _, _, n, _, _, _ = load_cfg(args.task)


epochs = []
train = []
test = []
for i in range(args.number):
    epochs.append(CHECKPOINT*i)
    _, loss = load_checkpoint(None, None, args.task, epoch=CHECKPOINT*i)
    train.append(loss['train'])
    test.append(loss['test'])

fig = draw_losses(epochs, train, test)

fig.savefig(args.file, dpi=200)
