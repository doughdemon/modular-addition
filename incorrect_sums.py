import argparse

import torch

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
    epoch, loss = load_checkpoint(model, None, args.task, epoch=args.epoch)
else:
    epoch, loss = load_checkpoint(model, None, args.task, final=True)

for j in range(n*n):
    a = j%n
    b = j//n
    c = (a + b) % n
    x = torch.tensor([[1.0 if k == a else 0.0 for k in range(n)], [1.0 if k == b else 0.0 for k in range(n)]]).transpose(0,1)
    y = model(x)
    if torch.max(y,0).indices != c:
         print(f"{a}+{b}:")
         print(f"Answer: {c} Prediction: {torch.max(y,0).indices}")
         print(y)
