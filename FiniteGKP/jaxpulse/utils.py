from jaxtyping import Array
import dynamiqs as dq
import jax.numpy as jnp

def soft_abs(x, min: float = .001):
    return jnp.sqrt(x**2 + min)

def heaviside(t: float):
    return 1.0*(t > 0)

def gaussian(mu, sig, t):
    sig = soft_abs(sig)
    return 1./(sig*jnp.sqrt(2*jnp.pi))*jnp.exp(-(t-mu)**2/(2*sig))

def commutator(A: Array, B: Array):
    return A@B - B@A

def anticommutator(A: Array, B: Array):
    return A@B + B@A

def rhodot_H(H: Array, rho: Array):
    return -1j*commutator(H, rho)

def dissipator(L: Array, rho: Array):
    return L@rho@dq.dag(L) - .5*anticommutator(L@dq.dag(L),rho)