import torch

# define the model

class MyModel(torch.nn.Module):
    def __init__(self, n, embed, hidden):
        super(MyModel, self).__init__()

        self.embed1 = torch.nn.Linear(n, embed, bias=False)
        self.embed2 = torch.nn.Linear(n, embed, bias=False)
        self.linear1 = torch.nn.Linear(2*embed, hidden, bias=False)
        self.unembed = torch.nn.Linear(hidden, n)

    def forward(self, x):
        a = self.embed1(x[:,0])
        b = self.embed2(x[:,1])
        x = torch.cat([a, b], dim=-1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.unembed(x)
        return x
