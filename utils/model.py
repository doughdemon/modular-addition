import torch

# define the model

def square(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        result = torch.square_(input)
    else:
        result = torch.square(input)
    return result

class MyModel(torch.nn.Module):
    def __init__(self, n, embed, hidden, activation="relu"):
        super(MyModel, self).__init__()

        self.embed1 = torch.nn.Linear(n, embed, bias=False)
        self.embed2 = torch.nn.Linear(n, embed, bias=False)
        self.linear1 = torch.nn.Linear(2*embed, hidden, bias=False)
        if activation == "relu":
            self.activation = torch.nn.functional.relu
        elif activation == "square":
            self.activation = square
        else: raise ValueError("recognized activation functions: 'relu', 'square'")
        self.unembed = torch.nn.Linear(hidden, n)

    def forward(self, x):
        a = self.embed1(x[:,0])
        b = self.embed2(x[:,1])
        x = torch.cat([a, b], dim=-1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.unembed(x)
        return x
