import argparse
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.checkpoints import *
from utils.config import *
from utils.figure_mpl import *
from utils.model import *
from utils.pca import *

parser = argparse.ArgumentParser(prog='make_pic.py', description='Visualize the first embedding')
parser.add_argument('task', help="task directory")
parser.add_argument('-f', '--file', default='animation.mp4')
parser.add_argument('-d', '--dims', type=int, default=2, help="number of PCA dimensions")

args = parser.parse_args()

assert args.dims in (2, 3)

CHECKPOINT = 100

_, _, layers, _, n, _, _, num_epochs = load_cfg(args.task)

FRAMES = num_epochs//CHECKPOINT

model = MyModel(n, layers['embed_dim'], layers['hidden_dim'])

Mlist = []
for i in range(FRAMES):
    load_checkpoint(model, None, args.task, epoch=i*PRINT)
    model.eval()

    M = pca(model.embed1.weight, dims=args.dims)

    Mlist.append(M.detach().numpy())

ani = draw_points_movie(Mlist, args.dims, [i*PRINT for i in range(FRAMES)])

FFwriter = animation.FFMpegWriter()
ani.save(args.file, writer=FFwriter)
