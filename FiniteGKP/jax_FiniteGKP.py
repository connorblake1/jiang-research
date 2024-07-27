import dynamiqs as dq
import qutip as qt
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from dynamiqs import tensor

run_tests = True

N = 170  # full computational
n = 25  # truncated for viewing / verification
hbar = 1
root2 = np.sqrt(2)
pi = np.pi

ket0 = dq.fock(2, 0)
ket1 = dq.fock(2, 1)
ketplus = (ket0 + ket1) / root2
ketminus = (ket0 - ket1) / root2

I_N = dq.eye(N)
a_aa = dq.destroy(N)
a_dag_aa = dq.create(N)
n_hat_aa = dq.number(N)
x_aa = (a_aa + a_dag_aa) / root2
p_aa = -1j * (a_aa - a_dag_aa) / root2

def D_aa(alpha_i):
    return jla.expm(alpha_i*a_dag_aa - np.conjugate(alpha_i)*a_aa)
def CD_aa(beta_i):
    return jla.expm(tensor((beta_i*a_dag_aa - np.conjugate(beta_i)*a_aa),dq.sigmaz())/(2*root2))
def S_aa(xi):
    return jla.expm(jnp.conj(xi)*(a_aa@a_aa)-xi*(a_dag_aa@a_dag_aa))
    
psi = dq.coherent(N, 0)
dq.plot_wigner(psi)

psi_shift = D_aa(2+1j)@psi
dq.plot_wigner(psi_shift)

l = 2 * np.sqrt(pi)
alpha = l * jnp.array([0, 1])
beta = l * jnp.array([1, 0])
l_j = jnp.array(
    [
        jnp.sqrt(alpha[0] ** 2 + beta[0] ** 2),
        jnp.sqrt(alpha[1] ** 2 + beta[1] ** 2),
    ]
)
q_j_aa = np.array(
    [
        alpha[0] * x_aa + beta[0] * p_aa,
        alpha[1] * x_aa + beta[1] * p_aa,
    ]
)
q_j_perp_aa = np.array(
    [
        alpha[0] * p_aa - beta[0] * x_aa,
        alpha[1] * p_aa - beta[1] * x_aa,
    ]
)
omega_12 = alpha[0] * beta[1] - beta[0] * alpha[1]
T_i_0_aa = jnp.array(
    [
        jla.expm(1j * q_j_aa[0]),
        jla.expm(1j * q_j_aa[1]),
    ]
)

