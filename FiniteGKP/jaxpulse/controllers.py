import equinox as eqx
from jaxtyping import Array
from abc import abstractmethod
import jax.numpy as jnp
from .utils import gaussian, soft_abs

__all__ = ["AbstractControl",
           "SinusoidalControl",
           "FrequencyControl",
           "GaussianControl",
           "GaussianShapedControl",
           "GaussianHeightControl",
           "PositiveGaussianControl",
           "GaussianPulseTrain",
           "GaussianShapedPulseTrain",
           "build_train",
           "ConstantControl",
           "ControlVector",
           "PeriodicControlVector"]

class AbstractControl(eqx.Module):

    @abstractmethod
    def __call__(self, t: float) -> float:
        pass

    def graph(self, times: Array, axhandle, label: str) -> None:
        values = jnp.array([self.__call__(time) for time in times])
        axhandle.plot(times, values,label=label)

class RealControl(AbstractControl):
    @abstractmethod
    def __call__(self, t: float) -> float:
        pass

class CompoundControl(AbstractControl):
    c1: RealControl
    c2: RealControl
    
    def __call__(self, t: float):
        return self.c1(t) + self.c2(t)



class SinusoidalControl(RealControl):
    a: Array
    omega: Array
    phi: Array
    def __call__(self, t: float) -> float:
        out = 0
        for i in range(len(self.a)):
            out = out + self.a[i]*jnp.sin(self.omega[i]*t + self.phi[i])
        return out

class FrequencyControl(SinusoidalControl):
    a: Array = eqx.field(static=True)
    omega: Array
    phi: Array = eqx.field(static=True)
    def __call__(self, t:float) ->float:
        return super().__call__(t)

class GaussianControl(RealControl):
    amp: Array
    mean: Array
    sigma: Array
    def std(amp: float, mean: float, sigma: float):
        return GaussianControl(
            amp=jnp.array([amp]),
            mean=jnp.array([mean]),
            sigma=jnp.array([sigma])
        )

    def __call__(self, t: float) -> float:
        out = 0
        for i in range(self.amp.shape[0]):
            out += self.amp[i]*gaussian(self.mean[i],self.sigma[i],t)
        return out
    def __add__(self, other):
        return GaussianControl(
            amp=jnp.hstack((self.amp, other.amp)),
            mean=jnp.hstack((self.mean, other.mean)),
            sigma=jnp.hstack((self.sigma, other.sigma))
        )
    def __mul__(self, other: float):
        return GaussianControl(
            amp=other*self.amp,
            mean=self.mean,
            sigma=self.sigma
        )

    
class GaussianShapedControl(GaussianControl):
    amp: Array
    mean: Array = eqx.field(static=True)
    sigma: Array = eqx.field(static=True)
    def __call__(self, t: float) -> float:
       return super().__call__(t)
    
class GaussianHeightControl(GaussianControl):
    amp: Array
    mean: Array = eqx.field(static=True)
    sigma: Array = eqx.field(static=True)
    def __call__(self, t: float) -> float:
        return super().__call__(t)
    
class PositiveGaussianControl(GaussianControl):
    def __call__(self, t: float) -> float:
        return soft_abs(super().__call__(t))
        
class GaussianPulseTrain(RealControl):
    # TODO subclass GaussianControl
    amp: Array
    mean: Array
    sigma: Array
    period: Array

    def std(amp: float, mean: float, sigma: float, period: float):
        return GaussianPulseTrain(
            amp=jnp.array([amp]),
            mean=jnp.array([mean]),
            sigma=jnp.array([sigma]),
            period=jnp.array([period])
        )
    
    def __add__(self, other):
        return GaussianPulseTrain(
            amp=jnp.hstack((self.amp, other.amp)),
            mean=jnp.hstack((self.mean, other.mean)),
            sigma=jnp.hstack((self.sigma, other.sigma)),
            period=jnp.hstack((self.period, other.period))
        )
    
    def __mul__(self, other: float):
        return GaussianPulseTrain(
            amp=other*self.amp,
            mean=self.mean,
            sigma=self.sigma,
            period=self.period
        )

    def __call__(self, t: float) -> float:
        out = 0
        for i in range(self.amp.shape[0]):
            t = jnp.remainder(t, self.period[i])
            out += self.amp[i]*gaussian(self.mean[i],self.sigma[i],t)
        return out
    
class GaussianShapedPulseTrain(GaussianPulseTrain):
    amp: Array
    mean: Array = eqx.field(static=True)
    sigma: Array = eqx.field(static=True)
    period: Array = eqx.field(static=True)
    def __call__(self, t: float) -> float:
        return super().__call__(t)
    
def build_train(gc: GaussianControl, period: float):
    if isinstance(gc, GaussianShapedControl):
        return GaussianShapedPulseTrain(
            amp=gc.amp,
            mean=gc.mean,
            sigma=gc.sigma,
            period=period
        )
    else:
        return GaussianPulseTrain(
            amp=gc.amp,
            mean=gc.mean,
            sigma=gc.sigma,
            period=period
        )
    
class ConstantControl(RealControl):
    k: float
    def __init__(self,k):
        self.k = k
    def __call__(self, t: float):
        return self.k


class ControlVector(eqx.Module):
    us: list[AbstractControl]

    def __call__(self, t: float) -> Array:
        return jnp.vstack((tuple([u(t) for u in self.us])))
    
    def __iter__(self):
        return iter(self.us)
    
    def __getitem__(self, index: int):
        return self.us[index]

class PeriodicControlVector(ControlVector):
    us: list[AbstractControl]
    period: Array
    def __init__(self, control: ControlVector, period: float):
        self.us = control.us
        self.period = jnp.array([period])

    def __call__(self, t: float) -> Array:
        return super().__call__(jnp.remainder(t,self.period))
    
