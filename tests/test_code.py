#%%
import jax 
jax.config.update("jax_enable_x64", True)
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from typing import NamedTuple, Callable, Container
import abc
from tqdm.auto import tqdm
from tqdm import trange
import optax
import matplotlib.pyplot as plt
from functools import partial
import diffrax as dfx
import jaxopt

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
    
    def spatial_diff_operator(self, func:Callable[[jnp.ndarray], float]): # u(x,y)-> u

        u_func = lambda x,y: jnp.sum(func(jnp.array([x,y])))

        ux = jax.grad(u_func, argnums=0)
        uxx = jax.grad(ux, argnums=0)
        uy = jax.grad(u_func, argnums=1)
        uyy = jax.grad(uy, argnums=1)
        v = self.params[0]
        Nx_func = lambda X: (uxx(X[0],X[1]) + uyy(X[0],X[1])) * v 
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

class NNconstructor(eqx.Module):
    param_restruct: Callable[[jnp.ndarray], jnp.ndarray]
    nn_static: eqx.Module
    filter_spec: Container # spec for time evolution

    def __call__(self, W):
        nn_param = self.param_restruct(W)
        nn = eqx.combine(nn_param, self.nn_static)
        return nn

    def get_w(self, nn):
        nn_param, nn_static = eqx.partition(nn, self.filter_spec)
        W, param_restruct = jax.flatten_util.ravel_pytree(nn_param)
        return W

class EvolutionalNN(eqx.Module):
    W: jnp.ndarray
    pde: PDE
    nnconstructor: NNconstructor
    get_N: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    get_J: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    get_gamma: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    
    @classmethod
    def from_nn(cls, nn, pde, filter_spec=eqx.is_array, gamma_method="inverse"):
        nn_param, nn_static = eqx.partition(nn, filter_spec)
        W, param_restruct = jax.flatten_util.ravel_pytree(nn_param)
        nnconstructor = NNconstructor(param_restruct, nn_static, filter_spec)

        @jax.jit
        def ufunc(W, x):
            _nn = nnconstructor(W)
            return _nn(x)   #def get_gamma(self, xs):
        
        Jf =  jax.jit(jax.jacfwd(ufunc, argnums=0))
        Jfv = jax.vmap(Jf, in_axes=(None, 0))

        @jax.jit
        def get_N(W, xs): #[batch, dim]
            # Spatial differential operator
            _nn = nnconstructor(W)
            nop = pde.spatial_diff_operator(_nn)
            return jax.vmap(nop)(xs)
        
        @jax.jit
        def get_J(W, xs):
            J = Jfv(W, xs)
            return J.reshape(xs.shape[0], len(W))

        # Define gamma method
        get_gamma = None
        if gamma_method == "inverse":
            @jax.jit        
            def get_gamma(W, xs):
                J = get_J(W, xs)
                N = get_N(W, xs)
                gamma = jnp.linalg.solve(J.T @ J, J.T @ N)
                return gamma
        elif gamma_method == "optimization": 
            @jax.jit
            def gamma_loss(gamma, J, N):
                errM = J.T @ J @ gamma - J.T @ N
                errs = errM.ravel()
                return jnp.mean(jnp.square(errs))

            @jax.jit        
            def get_gamma(W, xs, inital_gamma):
                J = get_J(W, xs)
                N = get_N(W, xs)
                loss = lambda gamma: gamma_loss(gamma, J, N)
                gamma = jaxopt.ScipyMinimize(loss, inital_gamma, method='L-BFGS-B')
                return gamma
                
        else: 
            raise ValueError("gamma_method must be either 'inverse' or 'optimization'")

        return cls(W, pde, nnconstructor, get_N, get_J, get_gamma)
    
    def new_w(self, W):
        return EvolutionalNN(W, self.pde, self.nnconstructor, self.get_N, self.get_J, self.get_gamma)

    def get_nn(self):
        return self.nnconstructor(self.W)
    
    def fit_initial(self, nbatch: int, nstep:int, optimizer, key: jr.PRNGKey, filter_spec=eqx.is_array, tol:float=1e-8):
        nn = self.get_nn()
        state = optimizer.init(eqx.filter(nn, filter_spec))
        sampler = Sampler(self.pde, nbatch)

        iter_step = trange(nstep)
        for i in iter_step:
            k_batch, key = jr.split(key)
            data = sampler.samp_init(k_batch) # sample initial function
            nn, state, loss = update_fn(nn, data, optimizer, state)
            iter_step.set_postfix({'loss':loss})
            if loss < tol:
                break
        W = self.nnconstructor.get_w(nn)
        return self.new_w(W)

    def ode (self, t,y, args):
        gamma = self.get_gamma(y, xs)
        return gamma


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

def plot2D(fig, ax, func, xspan=(0,1), yspan=(0,1), ngrid=100):
    x = jnp.linspace(*xspan, ngrid)
    y = jnp.linspace(*yspan, ngrid)
    X, Y = jnp.meshgrid(x, y)
    Z =  loop2d(X, Y, func)
    bar = ax.pcolor(X, Y, Z)
    fig.colorbar(bar, ax=ax)

# Setup PDE 
key = jr.PRNGKey(0)
pde = ParabolicPDE2D(jnp.array([1.]), jnp.array([[-jnp.pi, -jnp.pi], [jnp.pi, jnp.pi]]), jnp.array([0., 1.]))



# Learn initial condition
opt = optax.adam(learning_rate=optax.exponential_decay(1e-4, 3000, 0.9, end_value=1e-9))
nbatch = 10000


evonn = EvolutionalNN.from_nn(eqx.nn.MLP(2, 1, 20, 4, activation=jnp.tanh,key=key), pde, eqx.is_array)
evonnfit = evonn.fit_initial(nbatch, 10000, opt, key)
evonnfit.get_N(evonnfit.W, jr.normal(jr.PRNGKey(0), shape=(2,2)))
evonnfit.get_J(evonnfit.W, jr.normal(jr.PRNGKey(0), shape=(2,2)))

g = evonnfit.get_gamma(evonnfit.W, jr.normal(jr.PRNGKey(0), shape=(10,2)))
print(g)

# Evolve
xspans = pde.xspan.T
gen_xgrid = lambda xspan: jnp.linspace(xspan[0], xspan[1], 1)
xs_grids = jax.vmap(gen_xgrid)(xspans)
Xg = jnp.meshgrid(*xs_grids)
xs = jnp.stack([Xg[i].ravel() for i in range(len(Xg))]).T

def ode(t, y, args):
    gamma = evonnfit.get_gamma(y, xs)
    return gamma

term = dfx.ODETerm(ode)
solver = dfx.Euler()
saveat = dfx.SaveAt(ts=jnp.linspace(pde.tspan[0], pde.tspan[1], 100))
sol = dfx.diffeqsolve(term, solver, t0=pde.tspan[0], t1=pde.tspan[-1], dt0=0.1, y0=evonnfit.W, saveat=saveat)

print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
print(sol.ys)  # DeviceArray([1.   , 0.368, 0.135, 0.0498])
#%%

# Plotting
samp = Sampler(pde, nbatch)
data = samp.samp_init(key)
dinit = lambda x : - 2 * pde.params[0] * jnp.sin(x[0]) * jnp.sin(x[1])
dinit_pred = pde.spatial_diff_operator(evonnfit.get_nn())
dinit_diff = lambda x: dinit(x) - dinit_pred(x)
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(12,10))
axs = axes.ravel()
plot2D(fig, axs[0], pde.init_func, pde.xspan[:, 0], pde.xspan[:, 1], ngrid=100)
plot2D(fig, axs[1], evonnfit.get_nn(), pde.xspan[:, 0], pde.xspan[:, 1], ngrid=100)
plot2D(fig, axs[2], dinit, pde.xspan[:, 0], pde.xspan[:, 1], ngrid=100)
plot2D(fig, axs[3], dinit_pred, pde.xspan[:, 0], pde.xspan[:, 1], ngrid=100)
plot2D(fig, axs[4], dinit_diff, pde.xspan[:, 0], pde.xspan[:, 1], ngrid=100)

[a.set_title(t) for a, t in zip(axs, ["Initial Condition", "Predict Initial Condition", "N_x(u) at t =0", "Predict N_x(u) at t =0"])]
axs[0].scatter(data.x[:, 0], data.x[:, 1], s=0.1, label="sampled data")
axs[0].legend(loc='upper right')
[a.set_xlabel('x') for a in axs.ravel()];
[a.set_ylabel('y') for a in axs.ravel()];

# %%