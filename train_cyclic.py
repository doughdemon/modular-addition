import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random

from utils.checkpoints import *
from utils.config import *
from utils.model import *
from utils.pca import *

import matplotlib.pyplot as plt


PRINT = 100
CHECKPOINT = 100

task_dir = "tasks/first"

seed, frac_train, layers, lr, n, weight_decay, betas, num_epochs = load_cfg(task_dir)

def add(a: int, b: int) -> int:
    assert a >= 0 and a < n
    assert b >= 0 and b < n

    return (a+b)%n

# generate training and test data
# pick train samples at random from the multiplication table to form the training set
# use the whole multiplication table as the test set

test_x, test_y = [], []
for i in range(n*n):
    a = i%n
    b = i//n
    c = add(a, b)
    test_x.append([[1.0 if j == a else 0.0 for j in range(n)], [1.0 if j == b else 0.0 for j in range(n)]])
    test_y.append(c)

test_x = torch.tensor(test_x)
test_y = torch.tensor(test_y)

test_dataset = TensorDataset(test_x, test_y)
random.seed(seed)
train_dataset = torch.utils.data.Subset(test_dataset, random.sample(range(n*n), int(frac_train*n*n)))
train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))

lossfn = torch.nn.CrossEntropyLoss()

torch.manual_seed(1)#seed)
model = MyModel(n, layers['embed_dim'], layers['hidden_dim'])

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

load=False
if load:
    load_checkpoint(model, optimizer, task_dir, final=True)

# train the model

_iter = range(num_epochs)
try:
    import tqdm

    _iter = tqdm.tqdm(_iter)
except ImportError: pass

for i in _iter:
    for train_x, train_y in train_dataloader:
        optimizer.zero_grad()

        y = model(train_x)
        loss = lossfn(y, train_y)
        loss.backward()

        optimizer.step()

        if i % PRINT == 0:
            y = model(test_x)
            loss_test = lossfn(y, test_y)
            if i == num_epochs-1:
                for j in range(n*n):
                    if torch.max(y[j],0).indices != test_y[j]:
                        print(f"{j%n}+{j//n}:")
                        print(f"Answer: {test_y[j]} Prediction: {torch.max(y[j],0).indices}")
                        print(y[j])
            print(f"Epoch: {i} Training loss: {float(loss)} Test loss: {float(loss_test)}")

        if i % CHECKPOINT == 0:
            save_checkpoint(model, optimizer, task_dir, epoch=i)

save_checkpoint(model, optimizer, task_dir, final=True)

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

fig.savefig('MyFigure.png', dpi=200)
