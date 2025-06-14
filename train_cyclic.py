import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
import argparse

from utils.checkpoints import *
from utils.config import *
from utils.model import *
from utils.pca import *


PRINT = 100
CHECKPOINT = 100

parser = argparse.ArgumentParser(prog='train_cyclic.py', description='Train the neural net')
parser.add_argument('task', help="task directory")

args = parser.parse_args()

task_dir = args.task

seed, frac_train, layers, lr, n, weight_decay, betas, num_epochs = load_cfg(task_dir)

try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
except ImportError:
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

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

test_x = torch.tensor(test_x).to(device)
test_y = torch.tensor(test_y).to(device)

test_dataset = TensorDataset(test_x, test_y)
random.seed(seed)
train_dataset = torch.utils.data.Subset(test_dataset, random.sample(range(n*n), int(frac_train*n*n)))
train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))

lossfn = torch.nn.CrossEntropyLoss()

torch.manual_seed(seed)
model = MyModel(n, layers['embed_dim'], layers['hidden_dim']).to(device)

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

try:
    for i in _iter:
        for train_x, train_y in train_dataloader:
            optimizer.zero_grad()

            y = model(train_x)
            loss = lossfn(y, train_y)
            loss.backward()

            optimizer.step()

            if (i % PRINT)*(i % CHECKPOINT) == 0:
                with torch.no_grad():
                    y = model(test_x)
                    loss_test = lossfn(y, test_y)
                if i % PRINT == 0: print(f"Epoch: {i} Training loss: {float(loss)} Test loss: {float(loss_test)}")
                if i % CHECKPOINT == 0: save_checkpoint(model, optimizer, {'train': float(loss), 'test': float(loss_test)}, task_dir, epoch=i)
except KeyboardInterrupt:
    pass

save_checkpoint(model, optimizer, {'train': float(loss), 'test': float(loss_test)}, task_dir, final=True)
