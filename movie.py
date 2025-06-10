import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.checkpoints import *
from utils.config import *
from utils.model import *

PRINT = 100

task_dir = "tasks/first"

_, _, layers, _, n, _, _, num_epochs = load_cfg(task_dir)

FRAMES = num_epochs//PRINT

model = MyModel(n, layers['embed_dim'], layers['hidden_dim'])

Mlist = []
for i in range(FRAMES):
    load_checkpoint(model, None, task_dir, epoch=i*PRINT)
    model.eval()
#    torch.manual_seed(1)
#    A = model.embed1.weight.transpose(0,1)
#    V = torch.pca_lowrank(A)[2]
#    M = torch.matmul(A, V[:, :2])
    A = model.embed1.weight.transpose(0,1)
    A = A - A.mean(dim=0)
    U, S, _ = torch.linalg.svd(A)
    M = (U@torch.diag(S))[:, :2]
    Mlist.append(M.detach().numpy())
#    M.append(torch.load('data/pca' + str(i) + '.pt').detach().numpy())

fig, ax = plt.subplots()
scat = ax.scatter(Mlist[0][:,0], Mlist[0][:,1])
ax.set(xlim=(-2, 2), ylim=(-2, 2))

def update(frame):
    scat.set_offsets(Mlist[frame])
    return scat

ani = animation.FuncAnimation(fig, update, frames=FRAMES)

FFwriter = animation.FFMpegWriter()
ani.save('animation.mp4', writer=FFwriter)
