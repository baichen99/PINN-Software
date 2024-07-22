from NeuroPDE.config.common import CommonConfig
from NeuroPDE.train.pinn import PINN
from NeuroPDE.models.MLP import MLP
from NeuroPDE.pdes.gradients import gradients
import numpy as np

def pde(u, x, t):
    nu = 0.01 / np.pi
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_xx = gradients(u_x, x)
    return u_t + u*u_x - nu*u_xx

config = CommonConfig(
    epochs = 15000,
    val_freq = 1000,
    print_cols = ['epoch', 'loss', 'val_loss', 'pde_loss_0', 'bc_loss_0', 'l2_err_0'],
    learning_rate = 1e-3,
    net = [2, 20, 20, 20, 1],
    pde = pde,
    bc_weights=[100.0],
    pde_weights=[1.0],
    bc_data_path = "data/burgers/bc_data.csv",
    ic_data_path="data/burgers/ic_data.csv",
    test_data_path = "data/burgers/test_data.csv",
    pde_data_path = "data/burgers/pde_data.csv",
    X_dim = 2,
    U_dim = 1,
    
    # domain bounds
    lower_bound = [-1.0, 0.0],
    upper_bound = [1.0, 1.0],
    
    log_dir = f'logs/',
)


model = MLP(config.net, config.activation_fn).to(config.device)

PINN(config, model).train()
