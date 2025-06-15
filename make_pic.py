import argparse

import torch

from utils.checkpoints import *
from utils.config import *
from utils.figure_mpl import *
from utils.model import *
from utils.pca import *

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(prog='make_pic.py', description='Visualize the first embedding')
parser.add_argument('task', help="task directory")
parser.add_argument('-f', '--file', default='MyFigure.png')
parser.add_argument('-e', '--epoch', default=None)
parser.add_argument('-d', '--dims', type=int, default=2, help="number of PCA dimensions")

args = parser.parse_args()

assert args.dims in (2, 3)


_, _, layers, _, n, _, _, num_epochs = load_cfg(args.task)


model = MyModel(n, layers['embed_dim'], layers['hidden_dim'])
if args.epoch:
    epoch, loss = load_checkpoint(model, None, args.task, epoch=args.epoch)
else:
    epoch, loss = load_checkpoint(model, None, args.task, final=True)


M = pca(model.embed1.weight, dims=args.dims).detach().numpy()

fig = draw_points(M, args.dims, epoch, loss)

fig.savefig(args.file, dpi=200)
