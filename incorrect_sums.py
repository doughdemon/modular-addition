import argparse

import torch

from utils.accuracy import *
from utils.checkpoints import *
from utils.config import *
from utils.model import *

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(prog='make_pic.py', description='Show incorrect multiplications')
parser.add_argument('task', help="task directory")
parser.add_argument('-e', '--epoch', default=None)

args = parser.parse_args()


_, _, layers, _, n, _, _, _ = load_cfg(args.task)


model = MyModel(n, layers['embed_dim'], layers['hidden_dim'])
if args.epoch:
    load_checkpoint(model, None, args.task, epoch=args.epoch)
else:
    load_checkpoint(model, None, args.task, final=True)

acc, incorrect = accuracy(model, n)
print(f"Accuracy: {acc}")

for a,b,c,l in incorrect:
    print(f"{a}+{b}:")
    print(f"Answer: {c} Prediction: {l[0][1]}")
    print(l)
