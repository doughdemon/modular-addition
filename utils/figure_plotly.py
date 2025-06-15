import plotly.express as px

def draw_points(M, dims=2, epoch=None, loss=None):
    assert dims in (2, 3)

    X = M[:,0]
    Y = M[:,1]
    if dims == 3: Z = M[:,2]

    if dims == 2:
        fig = px.scatter(x=X, y=Y, text=range(len(X)))
    else:
        fig = px.scatter_3d(x=X, y=Y, z=Z, text=range(len(X)))

    return fig

def draw_lines(M, dims=2, epoch=None, loss=None, aut=1):
    assert dims in (2, 3)

    n = M.shape[0]

    idx = [(aut*i)%n for i in range(n)]

    M = M[idx,:]

    X = M[:,0]
    Y = M[:,1]
    if dims == 3: Z = M[:,2]

    if dims == 2:
        fig = px.line(x=X, y=Y, text=idx)
    else:
        fig = px.line_3d(x=X, y=Y, z=Z, text=idx)

    return fig
