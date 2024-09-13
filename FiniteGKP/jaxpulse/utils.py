from jaxtyping import Array
import dynamiqs as dq
import jax.numpy as jnp
import diffrax as dx
from typing import Callable
from jaxtyping import Array

def soft_abs(x, min: float = .0001):
    return jnp.sqrt(x**2 + min)

def heaviside(t: float):
    return 1.0*(t > 0)

def gaussian(mu, sig, t):
    sig = soft_abs(sig)
    return 1./(sig*jnp.sqrt(2*jnp.pi))*jnp.exp(-((t-mu)/sig)**2/(2))

def commutator(A: Array, B: Array):
    return A@B - B@A

def anticommutator(A: Array, B: Array):
    return A@B + B@A

def rhodot_H(H: Array, rho: Array):
    return -1j*commutator(H, rho)

def dissipator(L: Array, rho: Array):
    L_dag = dq.dag(L)
    return L@rho@L_dag - .5*anticommutator(L_dag@L,rho)

def diffrax_function_copy(
    f: Callable,
    t0: float,
    t1: float,
    dt: float
):
    ts = jnp.arange(t0,t1,dt)
    f_t = jnp.array([f(ti) for ti in ts])
    return dx.LinearInterpolation(ts=ts, ys=f_t).evaluate

def fun_from_array(
    ys: Array,
    t0: float,
    t1: float,
    dt: float
):
    return dx.LinearInterpolation(ts=jnp.arange(t0,t1,dt),ys=ys).evaluate
