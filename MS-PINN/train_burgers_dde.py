import deepxde as dde
import numpy as np
import pandas as pd 


def get_test_data():
    data = pd.read_csv("data/burgers/test_data.csv")
    X = data[['x', 't']].values
    y = data['u'].values[:, None]
    return X, y

def get_pde_data():
    data = pd.read_csv("data/burgers/pde_data.csv")
    X = data[['x', 't']].values
    return X

def get_bc_and_ic_data():
    bc_data = pd.read_csv("data/burgers/bc_data.csv")
    ic_data = pd.read_csv("data/burgers/ic_data.csv")
    # cat
    data = pd.concat([bc_data, ic_data])
    X = data[['x', 't']].values
    y = data['u'].values[:, None]
    return X, y

def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc_ic = dde.icbc.PointSetBC(*get_bc_and_ic_data())

data = dde.data.TimePDE(
    geomtime, pde, [bc_ic],
    anchors=get_pde_data()
)
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(iterations=10000)

# cal l2 error on test data
X, y_true = get_test_data()
y_pred = model.predict(X)
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))

