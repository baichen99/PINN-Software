from .gradients import gradients
import numpy as np

def Burgers(u, x, t):
    nu = 0.01 / np.pi
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_xx = gradients(u_x, x)
    return u_t + u*u_x - nu*u_xx

