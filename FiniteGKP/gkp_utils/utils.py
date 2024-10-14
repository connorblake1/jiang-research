import dynamiqs as dq
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
import strawberryfields as sf
root2 = jnp.sqrt(2.)

Delta = .2
GKP_N = 100
GKP_L = 2.*jnp.pi
alpha = GKP_L* jnp.array([0, 1],dtype=complex)
beta = GKP_L*jnp.array([-1,0],dtype=complex)

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

l_j = jnp.array(
    [
        jnp.sqrt(alpha[0] ** 2 + beta[0] ** 2),
        jnp.sqrt(alpha[1] ** 2 + beta[1] ** 2),
    ]
)
q_j = jnp.array(
    [
        alpha[0] * x + beta[0] * p,
        alpha[1] * x + beta[1] * p,
    ]
)
q_j_perp = jnp.array(
    [
        alpha[0] *p - beta[0] * x,
        alpha[1] * p - beta[1] * x,
    ]
)
omega_12 = alpha[0] * beta[1] - beta[0] * alpha[1]
T_j_0 = jnp.array(
    [
        jla.expm(1j * q_j[0]),
        jla.expm(1j * q_j[1]),
    ]
)
X_0 = jla.expm(1j*q_j[0]/2.)
Z_0 = jla.expm(1j*q_j[1]/2.)
Y_0 = jla.expm(1j*(q_j[0]+q_j[1])/2.)
x_j = jnp.array(
    [
        q_j[0]/l_j[0],
        q_j[1]/l_j[1]
    ] 
)
x_j_perp = jnp.array(
    [ 
        q_j_perp[0]/l_j[0],
        q_j_perp[1]/l_j[1]
    ]
)
x_j_m = jnp.asarray(np.load("fourier_saved.npy"))
c_Delta = jnp.cosh(Delta**2)
s_Delta = jnp.sinh(Delta**2)
t_Delta = jnp.tanh(Delta**2)
m_j = 2*jnp.pi/c_Delta/l_j
E_D = jla.expm(-Delta**2*n_hat)
E_D_plus = jla.expm(-Delta**2*(n_hat + IN))
E_D_minus = jla.expm(-Delta**2*(n_hat - IN))
E_D_inv = jla.inv(E_D)
c_n = .5*(E_D_minus@E_D_inv + E_D_plus@E_D_inv)
s_n = .5*(E_D_minus@E_D_inv - E_D_plus@E_D_inv)
T_j_E = jnp.array([E_D@T_j_0[0]@E_D_inv, E_D@T_j_0[1]@E_D_inv])
d_j_E = 1.0/root2*(x_j_m/jnp.sqrt(t_Delta) + 1j*x_j_perp*jnp.sqrt(t_Delta))
d_j_E_dag = np.array([jnp.conj(d_j.T) for d_j in d_j_E])
d_j_E_prod = np.array([d_j_E_dag[j]@d_j_E[j] for j in [0,1]])
X_E = jla.expm(.5*(1j*q_j[0]*c_Delta - q_j_perp[0]*s_Delta))
Z_E = jla.expm(.5*(1j*q_j[1]*c_Delta - q_j_perp[1]*s_Delta))
Y_E = 1j*Z_E@X_E
gamma = 1.  # free parameter
gamma_j = jnp.array([gamma,gamma])
epsilon_j = s_Delta*4*jnp.pi/l_j
theta_j = jnp.angle(alpha+1j*beta)
Gamma_dt = t_Delta/4*c_Delta**2*l_j**2

def R_x(theta):
    return jla.expm(-1j*theta*dq.sigmax()/2)
def R_z(theta):
    return jla.expm(-1j*theta*dq.sigmaz()/2)
def com(ai,bi):
    return ai@bi - bi@ai
def anticom(ai,bi):
    return ai@bi + bi@ai
@jax.jit
def sinm(ai):
    return -.5j*(jla.expm(1j*ai) - jla.expm(-1j*ai))
@jax.jit
def sawtooth_fourier(ai,mi,ni=30):
    # a is the matrix
    # ni is the fourier truncation
    # mi is the half-width of the pulse
    sum = jnp.zeros_like(ai)
    arg_a = ai*2*jnp.pi/mi
    for k in range(1,ni+1):
        sum = sum + ((-1)**k)/k*sinm(arg_a*k)
    return -mi/jnp.pi*sum
@jax.jit
def D(alpha_i: complex):
    return jla.expm(alpha_i*a_dag - jnp.conj(alpha_i)*a)
@jax.jit
def CD(beta_i: complex):
    return jla.expm(dq.tensor(beta_i*a_dag - jnp.conj(beta_i)*a,dq.sigmaz())/(2*root2))
@jax.jit
def S(xi):
    return jla.expm(jnp.conj(xi)*(a@a)-xi*(a_dag@a_dag))

# generate states
prog_gkp_fock = sf.Program(1)
with prog_gkp_fock.context as q:
    sf.ops.GKP(state=[0,0],epsilon=Delta**2) | q
eng = sf.Engine("fock", backend_options={"cutoff_dim": GKP_N, "hbar": 1})
logical_zero = jnp.array(eng.run(prog_gkp_fock).state.data).reshape((GKP_N,1))

prog_gkp_fock2 = sf.Program(1)
with prog_gkp_fock2.context as q:
    sf.ops.GKP(state=[np.pi,0],epsilon=Delta**2) | q
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


    