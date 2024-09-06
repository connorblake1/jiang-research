import equinox as eqx
from jaxtyping import Array
from abc import abstractmethod
import jax.numpy as jnp
from .utils import gaussian

__all__ = ["SinusoidalControl","FrequencyControl","GaussianControl","GaussianPulseTrain","GaussianShapeControl","GaussianHeightControl","ConstantControl","ControlVector"]

class AbstractControl(eqx.Module):

    @abstractmethod
    def __call__(self, t: float) -> float:
        pass

    def graph(self, times: Array, axhandle, label: str) -> None:
        values = jnp.array([self.__call__(time) for time in times])
        axhandle.plot(times, values,label=label)


class SinusoidalControl(AbstractControl):
    a: Array
    omega: Array
    phi: Array
    def __call__(self, t: float) -> float:
        out = 0
        for i in range(len(self.a)):
            out = out + self.a[i]*jnp.sin(self.omega[i]*t + self.phi[i])
        return out

class FrequencyControl(AbstractControl):
    omega: Array
    def __call__(self, t:float) ->float:
        out = 0
        for i in range(len(self.omega)):
            out = out + jnp.sin(self.omega[i]*t)
        return out

class GaussianControl(AbstractControl):
    amp: Array
    mean: Array
    sigma: Array
    def __call__(self, t: float) -> float:
        out = 0
        for i in range(self.amp.shape[0]):
            out += self.amp[i]*gaussian(self.mean[i],self.sigma[i],t)
        return out
    
class GaussianShapeControl(GaussianControl):
    amp: Array
    mean: Array = eqx.field(static=True)
    sigma: Array
    def __call__(self, t: float) -> float:
       return super().__call__(t)
    
class GaussianHeightControl(GaussianControl):
    amp: Array
    mean: Array = eqx.field(static=True)
    sigma: Array = eqx.field(static=True)
    def __call__(self, t: float) -> float:
        return super().__call__(t)
        
class GaussianPulseTrain(AbstractControl):
    # TODO subclass GaussianControl
    amp: Array
    mean: Array
    sigma: Array
    period: Array
    def __call__(self, t: float) -> float:
        t = jnp.remainder(t, self.period)
        out = 0
        for i in range(self.amp.shape[0]):
            out += self.amp[i]*gaussian(self.mean[i],self.sigma[i],t)
        return out
    
class ConstantControl(AbstractControl):
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

