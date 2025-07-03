import math
import argparse

import torch

from utils.config import *
from utils.figure_mpl import *
from utils.pca import *

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog='test_pca.py', description='Test the PCA')
parser.add_argument('task', help="task directory")
parser.add_argument('-f', '--file', default='test_pca.png')

args = parser.parse_args()

_, _, layers, _, n, _, _, num_epochs = load_cfg(args.task)

embed = layers['embed_dim']
L = []
for i in range(n):
    L.append([math.cos(2*math.pi*i/n), math.sin(2*math.pi*i/n)] + [0.0]*(embed-2))
L = torch.tensor(L).transpose(0,1)

R = torch.randn(embed, embed)
L = R @ L

M, _ = pca(L)

X = []
Y = []
S = []
m = []
for i in range(n):
    X.append(float(M[i][0]))
    Y.append(float(M[i][1]))

# plot

fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(X, Y)

for i in range(n):
    ax.text(X[i], Y[i], str(i))

fig = draw_points(M)

fig.savefig(args.file, dpi=200)
