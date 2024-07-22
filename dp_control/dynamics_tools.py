
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import typing
from typing import Callable
import gymnasium
# TODO setup ruff

def rk4_ctrl(ti: float,xvec: np.array, uvec: np.array ,fun: Callable, dti: float):

    # TODO enable multistep uvec

    k1 = dti*fun(ti,xvec,uvec)
    k2 = dti*fun(ti,xvec+k1/2.,uvec)
    k3 = dti*fun(ti,xvec+k2/2.,uvec)
    k4 = dti*fun(ti,xvec+k3,uvec)
    xvec += 1./6*(k1+2*k2+2*k3+k4)
    return xvec

# TODO jitify
def prop_control(dxdt: Callable,policy: Callable,x0: np.array,tf: float,N: int,n: int):
  t = np.linspace(0,tf,N)
  xt = np.zeros((N,n))
  dti = t[1]-t[0]
  xt[0,:] = x0
  for i in range(1,N):
    xt[i,:] = rk4_ctrl(t[i],xt[i-1,:],policy(t,xt[i,:]),dxdt,dti)
  return xt

# TODO: write phase portrait plotter, phase portrait truncation tools

# TODO: Gym integrations
