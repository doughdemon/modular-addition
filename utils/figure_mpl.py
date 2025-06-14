import matplotlib.pyplot as plt

def draw_points(M, dims=2, epoch=None, loss=None):
    assert dims in (2, 3)

    X = M[:,0]
    Y = M[:,1]
    if dims == 3: Z = M[:,2]

    fig = plt.figure()

    title = ""
    if epoch: title += f"Epoch: {epoch} "
    if loss:
        title += f"Train loss: {loss['train']:.2} Test loss: {loss['test']:.2}"
    fig.suptitle(title)

    if dims == 2:
        ax = fig.add_subplot()

        ax.scatter(X, Y)

        for i in range(len(X)):
            ax.text(X[i], Y[i], str(i))
    else:
        ax = fig.add_subplot(projection='3d')

        ax.scatter(X, Y, Z)

        for i in range(len(X)):
            ax.text(X[i], Y[i], Z[i], str(i))

    return fig


def draw_losses(epochs, train, test):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(epochs, train)
    ax.plot(epochs, test)

    return fig
