import argparse

import torch

from utils.checkpoints import *
from utils.config import *
from utils.model import *
from utils.pca import *

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(prog='make_pic.py', description='Create a figure')
parser.add_argument('task', help="task directory")
parser.add_argument('-f', '--file', default='MyFigure.png')
parser.add_argument('-d', '--dims', type=int, default=2, help="number of PCA dimensions")

args = parser.parse_args()

assert args.dims in (2, 3)


_, _, layers, _, n, _, _, num_epochs = load_cfg(args.task)


model = MyModel(n, layers['embed_dim'], layers['hidden_dim'])
load_checkpoint(model, None, args.task, final=True)


M = pca(model.embed1.weight, dims=3).detach().numpy()

X = M[:,0]
Y = M[:,1]
Z = M[:,2]

fig = plt.figure()
if args.dims == 2:
    ax = fig.add_subplot()

    ax.scatter(X, Y)

    for i in range(n):
        ax.text(X[i], Y[i], str(i))
else:
    ax = fig.add_subplot(projection='3d')

    ax.scatter(X, Y, Z)

    for i in range(n):
        ax.text(X[i], Y[i], Z[i], str(i))

fig.savefig(args.file, dpi=200)

