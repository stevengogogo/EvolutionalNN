#%%
import jax 
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from typing import NamedTuple, Callable, Container
import abc
from tqdm.auto import tqdm
from tqdm import trange
import optax
import matplotlib.pyplot as plt

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


class Data(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray

class PDE(eqx.Module):
    params: jnp.ndarray # parameters of the PDE
    xspan: jnp.ndarray # spatial domain
    tspan: jnp.ndarray # time domain
    @abc.abstractmethod
    def init_func(self, x):
        raise NotImplementedError
    @abc.abstractmethod
    def boundary_func(self, x):
        raise NotImplementedError
    @abc.abstractmethod
    def spatial_diff_operator(self, u_func): # u(x, )
        raise NotImplementedError

class ParabolicPDE2D(PDE):
    params: jnp.ndarray #[v]
    xspan: jnp.ndarray # spatial domain [[x_low, y_low], [x_high, y_high]]
    tspan: jnp.ndarray # time domain
    
    def init_func(self, x):
        return jnp.sin(x[0])* jnp.sin(x[1])
    
    def boundary_func(self, x):
        return jnp.array(0.)
    
    def spatial_diff_operator(self, u_func:Callable[[float,float], float]): # u(x,y)-> u
        ux = jax.grad(u_func, argnums=0)
        uxx = jax.grad(ux, argnums=0)
        uy = jax.grad(u_func, argnums=1)
        uyy = jax.grad(uy, argnums=1)
        v = self.params[0]
        Nx_func = lambda x: (uxx(x) + uyy(x)) * v 
        return Nx_func

class Sampler(eqx.Module):
    pde: PDE
    batch: int
    samp_init: Callable[[jr.PRNGKey], Data]
    def __init__(self, pde, batch):
        self.pde = pde
        self.batch = batch
        dim = pde.xspan.shape[1]
        
        def samp_init(key):
            x = jr.uniform(key, (batch, dim), minval=self.pde.xspan[0], maxval=self.pde.xspan[1])
            y = jax.vmap(self.pde.init_func)(x)
            return Data(x,y)
        
        self.samp_init = jax.jit(samp_init)

class EvolutionalNN(eqx.Module):
    nn: eqx.Module
    pde: PDE
    filter_spec: Container # spec for time evolution


    @staticmethod
    def get_w(nn, filter_spec):
        nn_param, nn_static = eqx.partition(nn, filter_spec)
        W, param_restruct = jax.flatten_util.ravel_pytree(nn_param)
        return W, param_restruct, nn_static
    
    def fit_initial(self, nbatch: int, nstep:int, optimizer, key: jr.PRNGKey, tol:float=1e-8):
        nn = self.nn 
        state = optimizer.init(eqx.filter(nn, eqx.is_array))
        sampler = Sampler(self.pde, nbatch)

        iter_step = trange(nstep)
        for i in iter_step:
            k_batch, key = jr.split(key)
            data = sampler.samp_init(k_batch) # sample initial function
            nn, state, loss = update_fn(nn, data, optimizer, state)
            iter_step.set_postfix({'loss':loss})
            if loss < tol:
                break

        return EvolutionalNN(nn, self.pde, self.filter_spec)

@eqx.filter_jit
def update_fn(nn: eqx.Module, data:Data, optimizer, state):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(nn, data)
    updates, new_state = optimizer.update(grad, state, nn)
    new_nn = eqx.apply_updates(nn, updates)
    return new_nn, new_state, loss

def loss_fn(nn, data:Data):
    y_preds = jax.vmap(nn)(data.x)
    return jnp.mean(jnp.square(y_preds.ravel() - data.y.ravel()))


@eqx.filter_jit
def loop2d(arr1, arr2, fun):
    funcex = jax.jit(lambda x,y: fun(jnp.stack([x,y])))
    fj = jax.vmap(funcex, in_axes=(0,0))
    fi = jax.vmap(fj, in_axes=(0,0))
    return fi(arr1, arr2).reshape(arr1.shape)

def plot2D(ax, func, xspan=(0,1), yspan=(0,1), ngrid=100):
    x = jnp.linspace(*xspan, ngrid)
    y = jnp.linspace(*yspan, ngrid)
    X, Y = jnp.meshgrid(x, y)
    Z =  loop2d(X, Y, func)
    ax.pcolor(X, Y, Z)
    return ax

# Setup PDE 
key = jr.PRNGKey(0)
pde = ParabolicPDE2D(jnp.array([1.]), jnp.array([[-jnp.pi, -jnp.pi], [jnp.pi, jnp.pi]]), jnp.array([0., 1.]))


# Learn initial condition
opt = optax.adam(1e-4)
nbatch = 300
evonn = EvolutionalNN(eqx.nn.MLP(2, 1, 30, 5, key=key), pde, eqx.is_array)
evonnfit = evonn.fit_initial(nbatch, 16000, opt, key)

#%%
# Plotting
samp = Sampler(pde, nbatch)
data = samp.samp_init(key)
fig, axs = plt.subplots(ncols=2, figsize=(10,5))
plot2D(axs[1], evonnfit.nn, pde.xspan[:, 0], pde.xspan[:, 1], ngrid=100)
plot2D(axs[0], pde.init_func, pde.xspan[:, 0], pde.xspan[:, 1], ngrid=100)
axs[0].set_title('Initial Condition')   
axs[1].set_title('Predict Initial Condition')
axs[0].scatter(data.x[:, 0], data.x[:, 1], s=0.1, label="sampled data")
axs[0].legend(loc='upper right')
[a.set_xlabel('x') for a in axs.ravel()]
[a.set_ylabel('y') for a in axs.ravel()]

# %%
