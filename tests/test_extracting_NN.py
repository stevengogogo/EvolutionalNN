"""
Extract neural network parameters and statics
"""
#%%
import jax 
jax.config.update("jax_enable_x64", True)
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

nn =  eqx.nn.MLP(1, 2, 2, 2, key=jr.PRNGKey(0))

# 
nn(jnp.array([1.]))

nn_param, nn_static = eqx.partition(nn, eqx.is_array)

nn_p_arr, restruct = jax.flatten_util.ravel_pytree(nn_param)

def u_nn(nn_p_arr, x):
    _nn_param = restruct(nn_p_arr)
    _nn = eqx.combine(_nn_param, nn_static)
    return _nn(x)


u_nn(nn_p_arr, jnp.array([1.]))

J = jax.jacfwd(u_nn)(nn_p_arr, jnp.array([1.])) # %%