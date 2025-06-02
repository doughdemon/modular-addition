import torch
import random

import matplotlib.pyplot as plt

n = 15 # cyclic group of order 2*n
embed = 2*n+1
hidden = 2*embed
train = int(.6*(2*n)*(2*n)) # data points for training
EPOCHS = 10000
PRINT = 100

# group multiplication in the dihedral group
# integers 0,...,n-1 represent the rotations
# integers n,...,2n-1 represent reflection times rotation
def add(a: int, b: int) -> int:
    assert a >= 0 and a < 2*n
    assert b >= 0 and b < 2*n

    return (a+b)%(2*n)
    s = n*((a // n + b // n) % 2)
    if b >= n: r = (n - a + b)%n
    else: r = (a + b)%n
    return r + s

# generate training and test data
# pick train samples at random from the multiplication table to form the training set
# use the whole multiplication table as the test set

train_a, train_b, train_c = [], [], []
test_a, test_b, test_c = [], [], []
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

load=False
if load:
    train_a = torch.load('data/traina.pt')
    train_b = torch.load('data/trainb.pt')
    train_c = torch.load('data/trainc.pt')
    model = torch.load('data/model.pt')
else:
    torch.save(train_a, 'data/traina.pt')
    torch.save(train_b, 'data/trainb.pt')
    torch.save(train_c, 'data/trainc.pt')

#optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=0.001, betas=(.9, .98))
#optimizer = torch.optim.SGD(model.parameters(), lr=.1, weight_decay=0.001)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
lossfn = torch.nn.CrossEntropyLoss()

# train the model

for i in range(EPOCHS):
    optimizer.zero_grad()

    y = model(train_a, train_b)
    loss = lossfn(y, train_c)
    loss.backward()

    optimizer.step()

    if (i+1) % PRINT == 0:
        y = model(test_a, test_b)
        loss_test = lossfn(y, test_c)
        print("epoch: {} training loss: {} test loss: {}".format(i, float(loss), float(loss_test)))
        torch.manual_seed(0)
        A = model.embed1.weight.transpose(0,1)
        V = torch.pca_lowrank(A)[2]
        M = torch.matmul(A, V[:, :2])
        torch.save(M, 'data/pca' + str(i//(PRINT)) + '.pt')
#        print(M)
#        scheduler.step()

torch.save(model, 'data/model.pt')

A = model.embed1.weight.transpose(0,1)
V = torch.pca_lowrank(A)[2]
M = torch.matmul(A, V[:, :2])

X = []
Y = []
S = []
m = []
for i in range(2*n):
    X.append(float(M[i][0]))
    Y.append(float(M[i][1]))

# plot

fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(X, Y)

for i in range(2*n):
    ax.text(X[i], Y[i], str(i))

fig.savefig('MyFigure.png', dpi=200)
