import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 15 # cyclic group of order 2*n
EPOCHS = 10000
PRINT = 100

FRAMES = EPOCHS//PRINT

M = []
for i in range(FRAMES):
    M.append(torch.load('data/pca' + str(i) + '.pt').detach().numpy())

fig, ax = plt.subplots()
scat = ax.scatter(M[0][:,0], M[0][:,1])
ax.set(xlim=(-2, 2), ylim=(-2, 2))

def update(frame):
    scat.set_offsets(M[frame])
    return scat

ani = animation.FuncAnimation(fig, update, frames=FRAMES)

FFwriter = animation.FFMpegWriter()
ani.save('animation.mp4', writer=FFwriter)
