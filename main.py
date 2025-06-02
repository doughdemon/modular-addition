import torch
import random

import matplotlib.pyplot as plt

n = 11 # order of the dihedral group
embed = 2*n+1
hidden = 2*embed
train = int(.9*(2*n)*(2*n)) # data points for training
EPOCHS = 10000
#0

# group multiplication in the dihedral group
# integers 0,...,n-1 represent the rotations
# integers n,...,2n-1 represent reflection times rotation
def add(a: int, b: int) -> int:
    assert a >= 0 and a < 2*n
    assert b >= 0 and b < 2*n

    s = n*((a // n + b // n) % 2)
    if b >= n: r = (n - a + b)%n
    else: r = (a + b)%n
    return r + s

# generate training and test data
# pick train samples at random from the multiplication table to form the training set
# use the whole multiplication table as the test set

train_a = []
train_b = []
train_c = []
test_a = []
test_b = []
test_c = []
rest = train
for i in range((2*n)*(2*n)):
    a = i%(2*n)
    b = i//(2*n)
    c = add(a, b)
    test_a.append([1.0 if j == a else 0.0 for j in range(2*n)])
    test_b.append([1.0 if j == b else 0.0 for j in range(2*n)])
    test_c.append(c)
    if random.random() <= rest/((2*n)*(2*n)-i):
        train_a.append([1.0 if j == a else 0.0 for j in range(2*n)])
        train_b.append([1.0 if j == b else 0.0 for j in range(2*n)])
        train_c.append(c)
        rest -= 1
train_a = torch.tensor(train_a)
train_b = torch.tensor(train_b)
train_c = torch.tensor(train_c)
test_a = torch.tensor(test_a)
test_b = torch.tensor(test_b)
test_c = torch.tensor(test_c)


# define the model

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.embed1 = torch.nn.Linear(2*n, embed)
        self.embed2 = torch.nn.Linear(2*n, embed)
        self.linear1 = torch.nn.Linear(2*embed, hidden)
        self.unembed = torch.nn.Linear(hidden, 2*n)

    def forward(self, a, b):
        a = self.embed1(a)
        b = self.embed2(b)
        x = torch.cat([a, b], dim=-1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.unembed(x)
        return x

model = MyModel()

optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
lossfn = torch.nn.CrossEntropyLoss()

# train the model

for i in range(EPOCHS):
    optimizer.zero_grad()

    y = model(train_a, train_b)
    loss = lossfn(y, train_c)
    loss.backward()

    optimizer.step()

    if (i+1) % 1000 == 0:
        y = model(test_a, test_b)
        loss_test = lossfn(y, test_c)
        print("epoch: {} training loss: {} test loss: {}".format(i, float(loss), float(loss_test)))


A = model.embed1.weight.transpose(0,1)
V = torch.pca_lowrank(A)[2]
M = torch.matmul(A, V[:, :3])

X = []
Y = []
Z = []
S = []
m = []
for i in range(2*n):
    X.append(float(M[i][0]))
    Y.append(float(M[i][1]))
    Z.append(float(M[i][2]))

# plot

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X, Y, Z)

for i in range(2*n):
    ax.text(X[i], Y[i], Z[i], str(i))

fig.savefig('MyFigure.png', dpi=200)
