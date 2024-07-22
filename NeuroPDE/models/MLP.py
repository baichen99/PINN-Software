import torch
from torch import nn

class MLP(torch.nn.Module):
    def __init__(self, seq_net, act: nn.Module, norm=False):
        # seq_net: [3, 20, 1]
        super(MLP, self).__init__()
        self.seq_net = seq_net
        self.num_layers = len(seq_net)
        self.layers = torch.nn.ModuleList()
        # last layer doesn't need activation
        for i in range(self.num_layers - 1):
            self.layers.append(torch.nn.Linear(seq_net[i], seq_net[i+1]))
        self.act = act()
        self.init_weights()
    def init_weights(self):
        for layer in self.layers:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)            


class MLPWithFFE(nn.Module):
    def __init__(self, seq_net, act: nn.Module, sigmas_x, sigmas_t, device='cuda'):
        super(MLPWithFFE, self).__init__()
        # [3, 20, 20, 3]
        self.net1 = MLP(seq_net[:-1], act).to(device)
        self.linear = MLP([seq_net[-2], seq_net[-1]], act).to(device)
        self.seq_net = seq_net

        # [1, 3]
        self.W_ts = [nn.Parameter(torch.randn(1, self.seq_net[0] // 2) * sigma).to(device) for sigma in sigmas_t]
        # # [2, 3]
        self.W_xs = [nn.Parameter(torch.randn(2, self.seq_net[0] // 2) * sigma).to(device) for sigma in sigmas_x]
        # self.W_ts = [(torch.randn(1, self.seq_net[0] // 2) * sigma).cuda() for sigma in sigmas_t]
        # self.W_xs = [(torch.randn(2, self.seq_net[0] // 2) * sigma).cuda() for sigma in sigmas_x]
    def forward(self, X):
        x = X[:, 0:2]
        t = X[:, 2:3]
        # (N, 1) @ (1, 3) => (N, 3) cat => (N, 6)
        H_ts = []
        H_xs = []
        for W_t in self.W_ts:
            H_ts.append(torch.cat([
                torch.sin(t @ W_t),
                torch.cos(t @ W_t),
            ], dim=1))
        
        # multiply
        H_ts_stack = torch.stack(H_ts)  # 将列表中的张量堆叠成一个新的张量
        H_t = torch.prod(H_ts_stack, dim=0)  # 计算逐元素乘积
        
        # (N, 2) @ (2, 3) => (N, 3) cat => (N, 6)
        for W_x in self.W_xs:
            H_xs.append(torch.cat([
                torch.sin(x @ W_x),
                torch.cos(x @ W_x),
            ], dim=1))
        
        # multiply
        H_xs_stack = torch.stack(H_xs)
        H_x = torch.prod(H_xs_stack, dim=0)
        
        # MLP
        H_x = self.net1(H_x)
        H_t = self.net1(H_t)
        # 对应元素相乘
        # 融合
        H = torch.multiply(H_t, H_x)
        out = self.linear(H) # => (N, 3)
        
        return out
    
if __name__ == '__main__':
    net = MLPWithFFE([6, 20, 20, 3], nn.Tanh(), sigmas_x=[1, 0.1, 0.01], sigmas_t=[1, 0.1, 0.01]).cuda()
    x = torch.ones((10, 3)).cuda()
    print(net(x).shape)
    # print(dict(net.named_parameters()))
    for param in net.parameters():
        print(param.shape)