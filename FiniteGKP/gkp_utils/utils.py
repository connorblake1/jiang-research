import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import strawberryfields as sf

GKP_DELTA = .2
GKP_N = 25
root2 = jnp.sqrt(2.)

# generate states
prog_gkp_fock = sf.Program(1)
with prog_gkp_fock.context as q:
    sf.ops.GKP(state=[0,0],epsilon=GKP_DELTA**2) | q
eng = sf.Engine("fock", backend_options={"cutoff_dim": GKP_N, "hbar": 1})
logical_zero = jnp.array(eng.run(prog_gkp_fock).state.data).reshape((GKP_N,1))

prog_gkp_fock2 = sf.Program(1)
with prog_gkp_fock2.context as q:
    sf.ops.GKP(state=[np.pi,0],epsilon=GKP_DELTA**2) | q
logical_one = jnp.array(eng.run(prog_gkp_fock2).state.data).reshape((GKP_N,1))
U = jnp.hstack((logical_zero,logical_one))
U_dag = dq.dag(U)
U_ident = U_dag@U
U_proj = U@U_dag


def Pi(rho):
    return U_proj@rho@U_proj
def Pi_perp(rho):
    return rho - Pi(rho)
def Pidot_2(sigma_dot):
    return .5*(dq.sigmax() * dq.trace(dq.sigmax() @ sigma_dot) +
            dq.sigmay() * dq.trace(dq.sigmay() @ sigma_dot) + 
            dq.sigmaz() * dq.trace(dq.sigmaz() @ sigma_dot))
def Pidot(rho_dot):
    return U@Pidot_2(U_dag@rho_dot@U)@U_dag
def Pidot_perp(rho_dot):
    return rho_dot - Pidot(rho_dot)
def sigma_proj(rho):
    return U_dag @ rho @ U
def psi_C(theta: float, phi: float):
    return jnp.cos(theta/2.)*logical_zero + jnp.exp(1j*phi)*jnp.sin(theta/2.)*logical_one
def rho_C(theta:float, phi: float):
    return dq.todm(psi_C(theta=theta,phi=phi))
def compute_bloch_vector(rho):
    sigma = sigma_proj(rho)
    return dq.expect([dq.sigmax(), dq.sigmay(), dq.sigmaz()], sigma)
def add_rho_to_bloch(rho, blocher):
    vec = compute_bloch_vector(rho)
    blocher.add_vectors(vec)


n_hat = dq.number(GKP_N)
a = dq.destroy(GKP_N)
a_dag = dq.create(GKP_N)
x = (a + a_dag) / root2
p = -1j * (a - a_dag) / root2

I2 = dq.eye(2)
IN = dq.eye(GKP_N)
II = dq.tensor(IN,I2)

NI = dq.tensor(n_hat,I2)
XI = dq.tensor(x, I2)
PI = dq.tensor(p, I2)

IZ = dq.tensor(IN,dq.sigmaz())
IX = dq.tensor(IN,dq.sigmax())
IY = dq.tensor(IN,dq.sigmay())
XZ = dq.tensor(x, dq.sigmaz())
PZ = dq.tensor(p, dq.sigmaz())

Ia = dq.tensor(IN, dq.destroy(2))
    