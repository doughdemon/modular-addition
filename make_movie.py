import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.checkpoints import *
from utils.config import *
from utils.model import *
from utils.pca import *

PRINT = 100

filename = 'animation.mp4'

task_dir = "tasks/first"

_, _, layers, _, n, _, _, num_epochs = load_cfg(task_dir)

FRAMES = num_epochs//PRINT

model = MyModel(n, layers['embed_dim'], layers['hidden_dim'])

Mlist = []
for i in range(FRAMES):
    load_checkpoint(model, None, task_dir, epoch=i*PRINT)
    model.eval()

    M = pca(model.embed1.weight)

    Mlist.append(M.detach().numpy())

fig, ax = plt.subplots()
scat = ax.scatter(Mlist[0][:,0], Mlist[0][:,1])
ax.set(xlim=(-2, 2), ylim=(-2, 2))

def update(frame):
    scat.set_offsets(Mlist[frame])
    return scat

ani = animation.FuncAnimation(fig, update, frames=FRAMES)

FFwriter = animation.FFMpegWriter()
ani.save(filename, writer=FFwriter)
