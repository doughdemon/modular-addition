import torch

from utils.checkpoints import *
from utils.config import *
from utils.model import *
from utils.pca import *

import matplotlib.pyplot as plt

filename = 'MyFigure.png'

task_dir = "tasks/first"

_, _, layers, _, n, _, _, num_epochs = load_cfg(task_dir)

model = MyModel(n, layers['embed_dim'], layers['hidden_dim'])

load_checkpoint(model, None, task_dir, final=True)

M = pca(model.embed1.weight)

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

fig.savefig(filename, dpi=200)
