from NeuroPDE.config.common import CommonConfig
from NeuroPDE.train.pinn import PINN
from NeuroPDE.models.PFNN import PFNN
from NeuroPDE.models.MLP import MLPWithFFE, MLP
from torch import nn
from NeuroPDE.pdes.gradients import gradients
import os


log_dir = f'logs/AgCu_Baseline/Temperature3'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(f'{log_dir}/checkpoints', exist_ok=True)

def pde(u, v, T, p, x, y, t):
    Cp, k = 448.0, 401.0
    rho = 8920
    mu = 0.0032
    T_x = gradients(T, x)
    T_y = gradients(T, y)
    T_xx = gradients(T_x, x)
    T_yy = gradients(T_y, y)
    
    h = Cp * T
    energy_time = rho * (gradients(h, t) + u * gradients(h, x) + v * gradients(h, y))
    heat_conduction = k * (T_xx + T_yy)
    energy_eq = energy_time - heat_conduction
    
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_y = gradients(u, y)
    
    
    v_t = gradients(v, t)
    v_x = gradients(v, x)
    v_y = gradients(v, y)
    
    u_xx = gradients(u_x, x)
    u_yy = gradients(u_y, y)

    v_xx = gradients(v_x, x)
    v_yy = gradients(v_y, y)
    
    p_x = gradients(p, x)
    p_y = gradients(p, y)
    
    # NS
    x_momentum = u_t + u*u_x + v*u_y - mu / rho * (u_xx + u_yy) + p_x / rho
    y_momentum = v_t + u*v_x + v*v_y - mu / rho * (v_xx + v_yy) + p_y / rho
    # continuity
    continuity = u_x + v_y
    
    return [x_momentum, y_momentum, continuity, energy_eq]



config = CommonConfig(
    epochs = 25000,
    val_freq = 200,
    learning_rate = 1e-3,
    lr_scheduler='cosine',
    # lr_decay = 0.1,
    # lr_decay_step = 5000,
    device = "cuda",
    net = [3] + [50] * 5 + [4],
    pde=pde,
    pde_weights=[10, 10, 1],
    bc_weights=[1000, 1000],
    bc_data_path = "data/AgCu/bc_data.csv",
    test_data_path = "data/AgCu/test_data.csv",
    pde_data_path = "data/AgCu/pde_data.csv",
    # domain
    lower_bound=[0, 0, 0],
    upper_bound=[0.05, 0.05, 5],
    
    adaptive_loss=True,

    RAR=True,
    RAR_num=5000,
    RAR_freq=1000,
    RAR_top_k=50,
    
    
    X_dim = 3, # x, y, t
    U_dim = 2, # u, v, T
    log_dir = log_dir,
    
    save_checkpoints=True,
    checkpoint_dir=f'{log_dir}/checkpoints',
    checkpoint_freq=200,
)


model = MLPWithFFE(config.net, nn.Tanh(), sigmas_x=[10, 1, 0.1, 0.01, 0.001, 0.0001], sigmas_t=[1, 0.1, 0.01]).to(config.device)

PINN(config, model).train()

