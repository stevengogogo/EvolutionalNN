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
    x_span: jnp.ndarray # spatial domain
    t_span: jnp.ndarray # time domain
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
    x_span: jnp.ndarray # spatial domain [[x_low, y_low], [x_high, y_high]]
    t_span: jnp.ndarray # time domain
    
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
    def __init__(self, pde, batch, key):
        self.pde = pde
        self.batch = batch
        dim = pde.x_span.shape[1]
        
        def samp_init(key):
            x = jr.uniform(key, (batch, dim), minval=self.pde.x_span[0], maxval=self.pde.x_span[1])
            y = self.pde.init_func(x)
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
    
    def fit_initial(self, nbatch: int, nstep:int, optimizer, key: jr.PRNGKey, tol:float=1e-4):
        nn = self.nn 
        state = optimizer.init(eqx.filter(nn, eqx.is_array))
        sampler = Sampler(self.pde, nbatch, key)

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
    return jnp.mean(jnp.square(y_preds - data.y))


# 
key = jr.PRNGKey(0)
pde = ParabolicPDE2D(jnp.array([0.1]), jnp.array([[0., 0.], [1., 1.]]), jnp.array([0., 1.]))
opt = optax.adam(1e-3)
evonn = EvolutionalNN(eqx.nn.MLP(2, 1, 2, 2, key=key), pde, eqx.is_array)

evonnfit = evonn.fit_initial(100, 400, opt, key)



# %%
