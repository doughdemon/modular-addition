import torch

def accuracy(model, n):
    incorrect = []
    for j in range(n*n):
        a = j%n
        b = j//n
        c = (a + b) % n
        x = torch.tensor([[1.0 if k == a else 0.0 for k in range(n)], [1.0 if k == b else 0.0 for k in range(n)]]).transpose(0,1)
        y = model(x)
        y = torch.nn.functional.softmax(y,dim=0)
        if y.max(0).indices != c:
            values,indices = y.sort(descending=True)
            values = [x.item() for x in values]
            indices = [x.item() for x in indices]

            incorrect.append((a,b,c, list(zip(indices,values))))

    return 1-len(incorrect)/(n*n), incorrect
