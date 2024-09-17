#%%
import jax 
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_debug_nans", True)
import equinox as eqx
import numpy as np
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
import numpy as np

class Data(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    dy:jnp.ndarray

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
    xspan: jnp.ndarray # spatial domain [[x_low, x_high], [y_low, y_high]]
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
    
    def u_true(self, x, t):
        # Analytical solution: sin(x)sin(y)exp(-2vt)
        return jnp.sin(x[0]) * jnp.sin(x[1]) * jnp.exp(-2 * self.params[0] * t)

class Sampler(eqx.Module):
    pde: PDE
    batch: int
    samp_init: Callable[[jr.PRNGKey], Data]
    def __init__(self, pde, batch):
        self.pde = pde
        self.batch = batch
        dim = pde.xspan.shape[0]
        dinit = pde.spatial_diff_operator(pde.init_func)
        def samp_init(key):
            x = jr.uniform(key, (batch, dim), minval=self.pde.xspan[:,0], maxval=self.pde.xspan[:,1])
            y = jax.vmap(self.pde.init_func)(x)
            dy = jax.vmap(dinit)(x)
            return Data(x,y, dy)
        
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
    def from_nn(cls, nn, pde, filter_spec=eqx.is_array):
        nn_param, nn_static = eqx.partition(nn, filter_spec)
        W, param_restruct = jax.flatten_util.ravel_pytree(nn_param)
        nnconstructor = NNconstructor(param_restruct, nn_static, filter_spec)

        @jax.jit
        def ufunc(W, xs):
            _nn = nnconstructor(W)
            u = lambda x: jnp.sum(_nn(x))
            us = jax.vmap(u)(xs)
            return us   # u(W, x)
        
        Jf =  jax.jacfwd(ufunc, argnums=0)

        @jax.jit
        def get_N(W, xs): #[batch, dim]
            # Spatial differential operator
            _nn = nnconstructor(W)
            nop = pde.spatial_diff_operator(_nn)
            return jax.vmap(nop)(xs)
        
        @jax.jit
        def get_J(W, xs):
            J = Jf(W, xs)
            return J

        # Define gamma method
        @jax.jit        
        def get_gamma(W, xs, tol=1e-5, **kwags):
            J = get_J(W, xs)
            N = get_N(W, xs)
            matvec = lambda x: jnp.dot(J.T @ J, x)
            gamma = jaxopt.linear_solve.solve_normal_cg(matvec, J.T @ N, tol=1e-5, **kwags)
            return gamma
            

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
            nn, state, loss = update_fn(nn, pde, data, optimizer, state)
            iter_step.set_postfix({'loss':loss})
            if loss < tol:
                break
        return self.from_nn(nn, self.pde, self.nnconstructor.filter_spec)

    def ode (self, t,y, args):
        #jax.debug.print("y : {y}", y=y)
        gamma = self.get_gamma(y, xs)
        #jax.debug.print("Gamma : {gamma}", gamma=gamma)
        return gamma

@eqx.filter_jit
def update_fn(nn: eqx.Module, pde, data:Data, optimizer, state):
    loss, grad = eqx.filter_value_and_grad(loss_pinn)(nn, pde, data)
    updates, new_state = optimizer.update(grad, state, nn)
    new_nn = eqx.apply_updates(nn, updates)
    return new_nn, new_state, loss

@eqx.filter_jit
def loss_pinn(nn, pde, data:Data):
    """
    loss = MSE(y- nn) + MSE(dy-dnn)
    """
    dnn = pde.spatial_diff_operator(nn)
    y_preds = jax.vmap(nn)(data.x)
    dy_preds = jax.vmap(dnn)(data.x)
    return mse(data.y, y_preds) + mse(data.dy, dy_preds)

@jax.jit
def mse(y, y_pred):
    return jnp.mean(jnp.square(y.ravel() - y_pred.ravel()))

class DrichletNN(eqx.Module):
    nn: eqx.Module 
    coeff: jnp.ndarray
    def __init__(self, nn):
        self.nn = nn
        self.coeff = jnp.array([1.])
    def __call__(self, x):
        L = 2 * jnp.pi
        omega = jnp.ones_like(x) * 2 * jnp.pi / L
        embed_v = self.fourier_embed(x, omega)
        return self.nn(embed_v)
    
    def fourier_embed(self, x, w):
        x_embed = x * w
        cos_embed = jax.vmap(jnp.cos)(x_embed)
        sin_embed = jax.vmap(jnp.sin)(x_embed)
        return jnp.concatenate([cos_embed, sin_embed], axis=-1)

# Setup PDE 
key = jr.PRNGKey(0)
pde = ParabolicPDE2D(jnp.array([1.]), jnp.array([[-jnp.pi, jnp.pi], [-jnp.pi, jnp.pi]]), jnp.array([0., 1.]))

#%%
# Learn initial condition
opt = optax.adam(learning_rate=optax.exponential_decay(1e-3, 2000, 0.9, end_value=1e-4))
nbatch = 5000
nn = DrichletNN(eqx.nn.MLP(2*2, 1, 30, 4, activation=jnp.tanh,key=key))
#nn = eqx.nn.MLP(2, 1, 30, 4, activation=jnp.tanh,key=key)
evonn = EvolutionalNN.from_nn(nn, pde)
evonnfit = evonn.fit_initial(nbatch, 10_000, opt, key)

#%%
xspans = pde.xspan
gen_xgrid = lambda xspan: jnp.linspace(xspan[0]+0.1, xspan[1]-0.1, 65)
xs_grids = jax.vmap(gen_xgrid)(xspans)
Xg = jnp.meshgrid(*xs_grids)
xs = jnp.stack([Xg[i].ravel() for i in range(len(Xg))]).T

evonnfit.get_N(evonnfit.W, xs)
evonnfit.get_J(evonnfit.W, xs)

g = evonnfit.get_gamma(evonnfit.W, xs)
print(g)

#%% Evolve
term = dfx.ODETerm(evonnfit.ode)
solver = dfx.Bosh3()
stepsize_controller = dfx.PIDController(rtol=1e-4, atol=1e-4)
saveat = dfx.SaveAt(ts=np.linspace(pde.tspan[0], pde.tspan[-1], 10).tolist())
sol = dfx.diffeqsolve(term, solver, t0=pde.tspan[0], t1=pde.tspan[-1], dt0=0.1, y0=evonnfit.W, saveat=saveat, stepsize_controller=stepsize_controller, progress_meter=dfx.TqdmProgressMeter(refresh_steps=2))
#print(sol.ys)  
#%%

# Plotting

# Boundaries
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
    bar = ax.pcolor(X, Y, Z, cmap='seismic')
    fig.colorbar(bar, ax=ax)



samp = Sampler(pde, nbatch)
data = samp.samp_init(key)
dinit = lambda x : - 2 * pde.params[0] * jnp.sin(x[0]) * jnp.sin(x[1])
dinit_pred = pde.spatial_diff_operator(evonnfit.get_nn())
dinit_diff = lambda x: dinit(x) - dinit_pred(x)
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(12,10))
axs = axes.ravel()
plot2D(fig, axs[0], pde.init_func, pde.xspan[0], pde.xspan[1], ngrid=100)
plot2D(fig, axs[1], evonnfit.get_nn(), pde.xspan[0], pde.xspan[1], ngrid=100)
plot2D(fig, axs[2], dinit, pde.xspan[0], pde.xspan[1], ngrid=100)
plot2D(fig, axs[3], dinit_pred, pde.xspan[0], pde.xspan[1], ngrid=100)
plot2D(fig, axs[4], dinit_diff, pde.xspan[0], pde.xspan[1], ngrid=100)

[a.set_title(t) for a, t in zip(axs, ["Initial Condition", "Predict Initial Condition", "N_x(u) at t =0", "Predict N_x(u) at t =0"])]
axs[0].scatter(data.x[:, 0], data.x[:, 1], s=0.1, label="sampled data")
axs[0].legend(loc='upper right')
[a.set_xlabel('x') for a in axs.ravel()];
[a.set_ylabel('y') for a in axs.ravel()];

# Plot comparison in 2D
i = 2 
w = sol.ys[i]
t = sol.ts[i]
fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax22 = ax2.ravel()
plot2D(fig2, ax22[0], lambda x: pde.u_true(x, t), pde.xspan[0], pde.xspan[1], ngrid=100)
plot2D(fig2, ax22[1], evonnfit.new_w(w).get_nn(), pde.xspan[0], pde.xspan[1], ngrid=100)
ax2[0].set_title(f"True N_x(u) at t = {t}")
ax2[1].set_title(f"Predict N_x(u) at t = {t}")
fig2.savefig("2dparabolic.png", dpi=300)

# Plot comparison in section
def plot_sections(ax, y, sol, evon, pde, u_true, label=None):
    xs = jnp.linspace(*pde.xspan[0], 100)
    for w, t in zip(sol.ys, sol.ts):

        nn = evonnfit.new_w(w).get_nn()
        u_trueF = lambda x: pde.u_true(jnp.array([x, y]), t)
        u_predF = lambda x: jnp.sum(nn(jnp.array([x, y])))

        u_true = jax.vmap(u_trueF)(xs)
        u_pred = jax.vmap(u_predF)(xs)
        ax3.plot(xs, u_true, color="black")
        ax3.plot(xs, u_pred, color="red", linestyle="--")

    ax3.plot(xs, u_true, color="black", label="Analytical Sol.")
    ax3.plot(xs, u_pred, color="red", linestyle="--", label="Prediction")
    ax3.legend()
    ax3.set_xlabel("x")
    ax3.set_ylabel("u")

fig3, ax3 = plt.subplots()
plot_sections(ax3, 1., sol, evonnfit, pde, pde.u_true)
fig3.savefig("1dparabolic.png", dpi=300)

# Error versus time
def plot_error(ax, sol, evon, pde, u_true, label=None):
    x = jnp.linspace(*pde.xspan[0], 100)
    y = jnp.linspace(*pde.xspan[1], 100)
    X, Y = jnp.meshgrid(x, y)
    XY = jnp.stack([X.ravel(), Y.ravel()]).T
    u_true0 = jax.vmap(pde.u_true, in_axes=(0,None))(XY, 0)
    err = []
    for w,t in zip(sol.ys, sol.ts):
        nn = evonnfit.new_w(w).get_nn()
        u_predf = lambda x: jnp.sum(nn(x))
        u_true = jax.vmap(pde.u_true, in_axes=(0,None))(XY, t)
        u_pred = jax.vmap(u_predf)(XY)
        error = jnp.linalg.norm(u_true - u_pred) / jnp.linalg.norm(u_true0)
        err.append(error)
    ax.plot(sol.ts, err, color="black", label=None)
    ax.set_xlabel("t")
    ax.set_ylabel("Relative Error")
    ax.set_yscale("log")
    return ax

fig4, ax4 = plt.subplots()
plot_error(ax4, sol, evonnfit, pde, pde.u_true)
fig4.savefig(f"error.png", dpi=300)

# %%
