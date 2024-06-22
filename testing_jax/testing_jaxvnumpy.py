import time
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
print(jax.devices())
N = [10,100,1000,10000]
np_times = []
jax_times = []
for n in N:
  print(n)
  main_diag = [i for i in range(n)]
  ui = [i for i in range(n-1)]
  uj = [i+1 for i in range(n-1)]
  li = uj
  lj = ui

  start_time = time.time()
  L = np.zeros((n,n))
  L[main_diag,main_diag] = -2
  L[ui,uj] = 1
  L[li,lj] = 1
  E,V = np.linalg.eigh(L)
  end_time = time.time()
  print("numpy:")
#print(L[:10,:10])
  np_times.append(end_time-start_time)

  start_time = time.time()
  Lj = jnp.zeros((n,n))
  Lj = Lj.at[main_diag,main_diag].set(-2)
  Lj = Lj.at[ui,uj].set(1)
  Lj = Lj.at[li,lj].set(1)
  E1,V1 = jnp.linalg.eigh(L)
  end_time = time.time()
  print("jax:")
# print(Lj[:10,:10])
  jax_times.append(end_time - start_time)
print(f"np: {np_times}")
print(f"jax: {jax_times}")
plt.plot(np.log10(np.array(N)),np_times)
plt.title("NUMPY")
plt.xlabel("log N")
plt.ylabel("Time (s)")
plt.savefig("numpy_scaling.png")
plt.clf()
plt.title("JAX")
plt.xlabel("log N")
plt.ylabel("Time (s)")
plt.plot(np.log10(np.array(N)),jax_times)
plt.savefig("jax_scaling.png")
