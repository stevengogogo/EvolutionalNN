"""
Jacobian of vector function
"""
#%%
import jax 
import jaxopt
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import flatten_util

din = 2
dout = 5
Nu = 100

nn =  eqx.nn.MLP(din, dout, 2, 2, key=jr.PRNGKey(0))


nn_param, nn_static = eqx.partition(nn, eqx.is_array)

Ws, restruct = flatten_util.ravel_pytree(nn_param)

Nw = len(Ws)

def u_nn(Ws, x):
    _nn_param = restruct(Ws)
    _nn = eqx.combine(_nn_param, nn_static)
    return jax.vmap(_nn)(x)

xs = jnp.ones((Nu, din))

# u_nn(Ws, x)

J = jax.jacfwd(u_nn, argnums=0)(Ws, xs).transpose(1, 0, 2)
Jt = J.transpose(0 , 2, 1)
gamma = jnp.ones((Nw,))
N = jnp.ones((Nu, dout))

assert J.shape == (dout, Nu, Nw)
assert Jt.shape == (dout, Nw, Nu)
assert (Jt@J).shape == (dout, Nw, Nw)
#J^T J gamma
assert (jnp.sum(Jt @ J, axis=0) @ gamma).shape == (Nw,)
# J^T N
assert (jnp.sum(jnp.sum(Jt @ N, axis=0), axis=-1)).shape == (Nw,)


A = jnp.sum(Jt @ J, axis=0) 
b = jnp.sum(jnp.sum(Jt @ N, axis=0), axis=-1)
def matvec_A(gamma):
  return  jnp.dot(A, gamma)

sol = jaxopt.linear_solve.solve_normal_cg(matvec_A, b, tol=1e-5)
assert sol.shape == (Nw,)
# %%
