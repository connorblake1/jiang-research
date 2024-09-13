# Auto-grad enabled BVP solver in JAX

import diffrax as dx
import jax.numpy as jnp
import jax.scipy.linalg as jla
import jax
import optax
from typing import Callable, Any
from jaxtyping import Array
import equinox as eqx

__all__ = ["QuantumBilinearController"]

class QuantumBilinearController(eqx.Module):
    F: Array
    Q: Array
    R: Array
    Ri: Array
    A: Array
    B: Array
    dA: Array
    H: Array
    Nt: int
    t0: float
    t1: float
    dt0: float
    n: int
    k: int
    solver: Any

    def __init__(self, F, Q, R, A, B, H, Nt, t1, t0 = 0.0, dt0= .1, solver = dx.Dopri5()):
        k = H.shape[2]
        n = H.shape[0]
        self.k = k
        self.n = n
        if not (
            F.shape == (n,n) and
            Q.shape == (n,n) and
            R.shape == (k,k) and
            A.shape == (n,n) and 
            B.shape == (n,k) and
            H.shape == (n,n,k)       
        ):  
            raise ValueError("One or more input matrices has the wrong shape.")
    
        self.F = F
        self.Q = Q
        self.R = R
        self.Ri = jnp.linalg.inv(R)
        self.A = A
        self.B = B
        self.dA = jnp.transpose(jnp.conj(A))
        self.H = H
        self.Nt = Nt
        self.t1 = t1
        self.t0 = t0
        self.dt0 = dt0
        self.solver = solver

    # TODO JIT everything, path optimize all einsums
    # @jax.jit
    def A_k_tilde(self, pk: Array):
        return self.A - .5 * (  jnp.einsum("ijk,kl,ml,m->ij",self.H,self.Ri,self.B,pk) + 
                                jnp.einsum("ir,rs,mjs,m->ij",self.B,self.Ri,self.H,pk))
    
    # @jax.jit
    def B_k_tilde(self, xk: Array):
         return self.B + jnp.einsum("jnl,n->jl",self.H,xk)

    # @jax.jit
    def S_k_tilde(self, xk: Array):
        B_k_tilde_t = self.B_k_tilde(xk)
        return (jnp.einsum("ik,kl,jl->ij",B_k_tilde_t,self.Ri,B_k_tilde_t) -
                .5*(jnp.einsum("iqp,q,pr,jr->ij",self.H,xk,self.Ri,self.B) + 
                    jnp.einsum("it,tv,jwv,w->ij",self.B,self.Ri,self.H,xk)))

    # @jax.jit
    def Q_k_tilde(self, pk: Array):
        return (jnp.einsum("m,mik,kl,njl,n->ij",pk,self.H,self.Ri,self.H,pk) +
                jnp.einsum("m,mjr,rs,nis,n->ij",pk,self.H,self.Ri,self.H,pk))

    # @jax.jit
    def A_k(self, xk: Array, pk: Array, Pk: Array):
        return self.A_k_tilde(pk) - self.S_k_tilde(xk) @ Pk
    
    # @jax.jit
    def Q_k(self, xk: Array, pk: Array, Pk: Array):
        return self.Q_k_tilde(pk) - jnp.einsum("im,mn,nj->ij",Pk,self.S_k_tilde(xk),Pk)

    # @jax.jit
    def u_k(self, xk: Array, xk1: Array, Pk1: Array):
        return jnp.einsum("kl,jnl,n,jm,m->k",-self.Ri,self.H,xk,Pk1,xk1)
    
    def x_prop(self, x0: Array, field: Callable[[float, Array, Any],Array]):
        return dx.diffeqsolve(
            terms=dx.ODETerm(field), # TODO speedup
            t0=self.t0,
            t1=self.t1,
            dt0=self.dt0,
            y0=x0,
            saveat=dx.SaveAt(dense=True),
            solver=self.solver,
            max_steps=self.Nt
        ).evaluate
    
    def compute_J(self, xt, ut, ts):
        Jf = .5*jnp.einsum("i,ij,j->",jnp.squeeze(xt[-1,:]),self.F,jnp.squeeze(xt[-1,:]))
        Jt = jax.vmap(lambda xti, uti: .5*jnp.einsum("i,ij,j->",jnp.squeeze(xti),self.Q,jnp.squeeze(xti)) + .5*jnp.einsum("k,kl,l->",uti,self.R,uti), in_axes=(0,0))
        Jt_integrate = jax.scipy.integrate.trapezoid(Jt(xt, ut), ts)
        print(f"{Jf=},{Jt_integrate=} Total J={Jt_integrate+Jf}")

    def solve(
        self,
        x0: Array,
        N_its: int = 5,
    ):
        ts = jnp.arange(self.t0, self.t1, self.dt0)
        Nt = ts.shape[0]
        x_res = jnp.zeros((N_its+1,Nt,self.n))
        u_res = jnp.zeros((N_its+1,Nt,self.k))

        # eq15b
        P_k_t = dx.diffeqsolve(
            terms=dx.ODETerm(lambda t, y, args: -y @ self.A - self.dA @ y - self.Q + 
                             jnp.einsum("im,mk,kl,nl,nj->ij",y,self.B,self.Ri,self.B,y)),
            t0=self.t1,
            t1=self.t0,
            dt0=-self.dt0,
            y0=self.F,
            saveat=dx.SaveAt(dense=True),
            solver=self.solver,
            max_steps=self.Nt
        ).evaluate
        # eq15b

        # eige, eigv = jnp.linalg.eigh(self.A) 
        def no_control(t: float, y: Array, args: Any):
            return self.A @ y
        def initial_field(t: float, y: Array, args: Any):
            Pkt = P_k_t(t)
            return (self.A - jnp.einsum("ik,kl,nl,nj->ij",self.B,self.Ri,self.B,Pkt)) @ y

        x_k_t = self.x_prop(x0=x0, field=no_control)
        x_0_t = jnp.array([x_k_t(ti) for ti in ts])
        x_res = x_res.at[0,:,:].set(x_0_t.reshape((Nt,-1)))

        self.compute_J(x_0_t, u_res[0,:,:],ts)

        x_k_t = self.x_prop(x0=x0, field=initial_field)
        x_1_t = jnp.array([x_k_t(ti) for ti in ts])
        x_res = x_res.at[1,:,:].set(x_1_t.reshape((Nt,-1)))
        u_1_t = jnp.array([- jnp.einsum("kl,nl,nj->kj",self.Ri,self.B,P_k_t(ti))@jnp.squeeze(x_k_t(ti)) for ti in ts])
        u_res = u_res.at[1,:,:].set(u_1_t.reshape((Nt,-1)))
        self.compute_J(x_1_t, u_1_t,ts)

        # iterative solve
        for i in range(1,N_its):
            # iterate step
            def A_k_t_field(t: float, y: Array, args: Any):
                Pkt = P_k_t(t)
                xkt = jnp.squeeze(x_k_t(t))
                pkt = Pkt@xkt
                return self.A_k(xk=xkt,pk=pkt,Pk=Pkt)@y
            def Pk1_field(t:float, y: Array, args: Any):
                Pkt = P_k_t(t)
                xkt = jnp.squeeze(x_k_t(t))
                pkt = Pkt@xkt
                Akt = self.A_k(xk=xkt,pk=pkt,Pk=Pkt)
                Qkt = self.Q_k(xk=xkt,pk=pkt,Pk=Pkt)
                return -jnp.einsum("im,mj->ij",y,Akt) - jnp.einsum("mi,mj->ij",Akt,y) - Qkt

            x_k1_t = self.x_prop(x0=x0, field=A_k_t_field)

            P_k1_t = dx.diffeqsolve(
                terms=dx.ODETerm(Pk1_field),
                t1=self.t0,
                t0=self.t1,
                dt0=-self.dt0,
                y0=self.F,
                saveat=dx.SaveAt(dense=True),
                solver=self.solver,
                max_steps=self.Nt
            ).evaluate

            u_k_t = dx.LinearInterpolation(
                ts=ts,
                ys=jnp.array(
                    [
                        self.u_k(
                            xk=jnp.squeeze(x_k_t(ti)),
                            xk1=jnp.squeeze(x_k1_t(ti)),
                            Pk1=P_k1_t(ti)
                        ) for ti in ts
                    ]
                )
            ).evaluate
            u_eval_t = jnp.array([u_k_t(ti) for ti in ts])


            # evaluate system
            def eval_field(t: float, y: Array, args: Any):
                Pkt = P_k_t(t)
                xkt = jnp.squeeze(x_k_t(t))
                pkt = Pkt@xkt
                ukt = u_k_t(t)
                return (self.A_k_tilde(pk=pkt)@y + 
                        jnp.reshape(self.B_k_tilde(xkt)@ukt,(-1,1)))
                        
            x_eval_t = dx.diffeqsolve(
                terms=dx.ODETerm(eval_field),
                t0=self.t0,
                t1=self.t1,
                dt0=self.dt0,
                y0=x0,
                saveat=dx.SaveAt(ts=ts),
                solver=self.solver,
                max_steps=self.Nt
            ).ys

            self.compute_J(x_eval_t, u_eval_t, ts)


            x_res = x_res.at[i+1,:,:].set(x_eval_t.reshape(Nt,-1))
            u_res = u_res.at[i+1,:,:].set(u_eval_t.reshape(Nt,-1))


            # copy functions for next iteration
            x_k_t = dx.LinearInterpolation(
                ts=ts,
                ys=jnp.array(
                    [
                        x_k1_t(ti)
                        for ti in ts
                    ]
                )
            ).evaluate

            P_k_t = dx.LinearInterpolation(
                ts=ts,
                ys=jnp.array(
                    [
                        P_k1_t(ti)
                        for ti in ts
                    ]
                )
            ).evaluate
        return x_res, u_res, ts
        


## TODO: Make composite interpolator class
    # allow to multiply, add, rebuild, etc



