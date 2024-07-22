from NeuroPDE.config.common import CommonConfig
from NeuroPDE.train.pinn import PINN
from NeuroPDE.models.MLP import MLP


config = CommonConfig(
    epochs = 5000,
    val_freq = 100,
    learning_rate = 1e-3,
    lr_decay = 0.1,
    lr_decay_step = 10000,
    net = [3] + [50] * 6 + [3],
    pde = 'InverseNS',
    
    bc_data_path = "data/cylinder_wake_ns/bc_data.csv",
    test_data_path = "data/cylinder_wake_ns/bc_data.csv",
    pde_data_path = "data/cylinder_wake_ns/pde_data.csv",
    
    X_dim = 3,
    U_dim = 2,
    
    # domain bounds
    lower_bound = [1, -2, 0],
    upper_bound = [8, 2, 7],
    
    # Residual-based adaptive refinement
    RAR = True,
    resample_freq = 1000,
    RAR_num=50000,
    RAR_k = 100,  # choose residual top k points to resample
    
    # loss weights
    pde_weights = [1.0],
    bc_weights = [1.0],
    
    activation = "tanh",
    optimizer = "Adam",
    
    params_init = [0.0, 0.0],

    log_dir = f'logs',
)


model = MLP(config.net, config.activation_fn).to(config.device)

PINN(config, model).train()

