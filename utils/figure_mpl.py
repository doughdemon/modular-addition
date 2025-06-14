import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def draw_points_movie(M, dims=2, epochs=None):
    assert dims in (2, 3)
    fig = plt.figure()

    if dims == 2:
        ax = fig.add_subplot()

        def update(frame):
            title=""
            if epochs: title += f"Epoch: {epochs[frame]}"
            fig.suptitle(title)

            ax.cla()
            ax.set(xlim=(-2, 2), ylim=(-2, 2))
            ax.scatter(M[frame][:,0], M[frame][:,1])

        ani = animation.FuncAnimation(fig, update, frames=len(M))
    else:
        ax = fig.add_subplot(projection="3d")

        def update(frame):
            title=""
            if epochs: title += f"Epoch: {epochs[frame]}"
            fig.suptitle(title)

            ax.cla()
            ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2,2))
            ax.scatter(M[frame][:,0], M[frame][:,1], M[frame][:,2])

        ani = animation.FuncAnimation(fig, update, frames=len(M))

    return ani


def draw_losses(epochs, train, test):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(epochs, train)
    ax.plot(epochs, test)

    return fig
