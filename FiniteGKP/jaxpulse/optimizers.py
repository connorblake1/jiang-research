import equinox as eqx
from jaxtyping import Array
from typing import Callable
import diffrax as df
import optax
from functools import partial
from abc import abstractmethod
import jax.numpy as jnp
import dynamiqs as dq
from .controllers import ControlVector
from .utils import dissipator, rhodot_H

__all__ = ["ClosedQuantumSystem", "OpenQuantumSystem", "OptimalController"]

class AbstractSystem(eqx.Module):
    H_0: Array
    H_M: list[Array]

    @abstractmethod
    def vector_field_term(self, t: float, y, args: dict, controls: ControlVector):
        pass
    def run_simulation(self, ts: Array, dt: float, y0: Array, u: ControlVector):
        results = df.diffeqsolve(
            terms=df.ODETerm(partial(self.vector_field_term, controls=u)),
            solver=df.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=dt,
            y0=y0,
            saveat=df.SaveAt(ts=ts),
            max_steps=1_000_000
        )
        return results.ys
        

class ClosedQuantumSystem(AbstractSystem):

    def vector_field_term(self, t: float, y: Array, args: dict, controls: ControlVector):
        field = self.H_0
        for u_m, H_m in zip(controls, self.H_M):
            field = field + H_m*u_m(t)
        return (-1j*field) @ y
    

class OpenQuantumSystem(AbstractSystem):
    U_K: list[Array]
    C_K: list[Array]

    def vector_field_term(self, t: float, y: Array, args: dict, controls: ControlVector):
        controls_H = controls[:len(self.H_M)]
        controls_L = controls[len(self.H_M):]
        # Hamiltonians
        H_tot = self.H_0 + sum([H_m*u_m(t) for u_m, H_m in zip(controls_H, self.H_M)])
        field = rhodot_H(H_tot, y)
        # Lindbladians
        for U_k in self.U_K:
            field = field + dissipator(U_k,y)
        for v_k, C_k in zip(controls_L, self.C_K):
            field = field + v_k(t)*dissipator(C_k,y)
        return field
    def run_simulation(self, ts: Array, dt: float, y0: Array, u: ControlVector):
        nonnormalized = super().run_simulation(ts, dt, y0, u)
        return nonnormalized/(dq.trace(nonnormalized).reshape((nonnormalized.shape[0],1,1)))
    
class OptimalController(eqx.Module):
    system: AbstractSystem
    controls: ControlVector
    y0: Array
    duration: float
    dt_start: float
    dt_save: float
    y_final: Callable[Array, float]
    y_statewise: Callable[[Array, Array, float], float]
    times: Array
    n_H_M: int
    n_C_K: int

    def __init__(self,
            system: AbstractSystem,
            controls: ControlVector,
            y0: Array,
            duration: float,
            y_final: Callable[Array, float],
            y_statewise:  Callable[[Array, Array, float], float] = lambda y, u, t: 0,
            dt_start: float = .01,
            dt_save: float = .1,
        ):
        self.system = system
        self.controls = controls
        self.n_H_M = len(self.system.H_M)
        if isinstance(self.system, ClosedQuantumSystem):
            self.n_C_K = 0
            if len(list(self.controls)) != self.n_H_M:
                raise ValueError("Incorrect number of control Hamiltonians or controls.")
        else:
            self.n_C_K = len(self.system.C_K)
            if len(list(self.controls)) != self.n_C_K + self.n_H_M:
                raise ValueError("Incorrect number of dissipator controls or controllable dissipators.")
        if y0.shape[0] == y0.shape[1] and isinstance(self.system, ClosedQuantumSystem):
            raise TypeError("Attempted to assign density operator in place of wave function.")
        elif y0.shape[1] == 1 and isinstance(self.system, OpenQuantumSystem):
            raise TypeError("Attempted to assign wave function in place of density operator.")
        self.y0 = y0
        self.duration = duration
        self.dt_save = dt_save
        self.dt_start = dt_start
        self.y_final = y_final
        self.y_statewise = y_statewise
        self.times = jnp.arange(0.0, self.duration, self.dt_save)

    def _run(self, controls: ControlVector):        
        return self.system.run_simulation(
            ts=self.times,
            dt=self.dt_start,
            y0=self.y0,
            u=controls
        )
    
    def run(self):
        return self._run(self.controls)

    def loss(self, controls):
        return jnp.real(self.penalty(self._run(controls)))

    def penalty(self, yt):
        # TODO jaxify
        L = 0.0
        for i in range(yt.shape[0]):
            ti = self.times[i]
            L += self.y_statewise(yt[i], self.controls(ti), ti)
        L += self.y_final(yt[-1])
        return L

    def optimize(
            self,
            N_steps: int,
            learning_rate: float = .1,
            verbosity: int = 0) -> None:
        optim = optax.adam(learning_rate, nesterov=True)
        opt_state = optim.init(self.controls)
        ctrls = self.controls
        
        @eqx.filter_jit
        def make_step(ctrls, opt_state):
            loss_val, grads = eqx.filter_value_and_grad(self.loss)(ctrls)
            updates, opt_state = optim.update(grads, opt_state)
            ctrls = eqx.apply_updates(ctrls, updates)
            return ctrls, opt_state, loss_val

        for step in range(N_steps):
            ctrls, opt_state, loss_val = make_step(ctrls, opt_state=opt_state)
            if verbosity > 0:
                if verbosity > 1:
                    eqx.tree_pprint(ctrls, short_arrays=False)
                print(f"Step: {step} Loss: {loss_val}")
            if loss_val < 0:
                break
        
        return OptimalController(
            system=self.system,
            controls=ctrls,
            y0=self.y0,
            duration=self.duration,
            dt_start=self.dt_start,
            dt_save=self.dt_save,
            y_final=self.y_final,
            y_statewise=self.y_statewise,
        )

    def plot(
            self,
            ax,
            exp_ops: list = [],
            exp_names: list[str] = [],
            plot_controls: bool = True
        ):
        yt = self.run()
        exps = dq.expect(exp_ops, yt)
        for i, name in enumerate(exp_names):
            ax.plot(self.times,jnp.real(exps[i]),label=name)
        if plot_controls:
            for i in range(self.n_H_M):
                u_m = self.controls[i]
                u_m.graph(self.times,ax,fr"$u_{i+1}(t)$")
            if isinstance(self.system, OpenQuantumSystem):
                for i in range(self.n_C_K):
                    v_i = self.controls[self.n_H_M+i]
                    v_i.graph(self.times,ax,fr"$v_{i+1}(t)$")
        


            