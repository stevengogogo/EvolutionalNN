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
from time import time
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


def u_true(x, t, a):
    return jnp.exp(-4*jnp.pi**2*a*t)*jnp.sin(2*jnp.pi*x) #+ 2/(jnp.pi**2 * a) * (1 - jnp.exp(-jnp.pi**2*a*t))*jnp.sin(jnp.pi*x)

#sigma = lambda x,t: 2*jnp.sin(jnp.pi*x)

class ParabolicPDE1D(PDE):
    params: jnp.ndarray #[v]
    xspan: jnp.ndarray # spatial domain [[x_low, x_high]]
    tspan: jnp.ndarray # time domain
    
    def init_func(self, x):
        return u_true(x, 0., self.params[0])
    
    def spatial_diff_operator(self, func:Callable[[jnp.ndarray], float]): # u(x,y)-> u

        u_func = lambda x: jnp.sum(func(jnp.array([x])))

        ux = jax.grad(u_func, argnums=0)
        uxx = jax.grad(ux, argnums=0)
        v = self.params[0]
        Nx_func = lambda X: self.params[0]*uxx(X[0]) #+ sigma(X[0],0) 
        return Nx_func

    def boundary_func(self, x):
        return jnp.array(0.)
    
    def u_true(self, x, t):
        # Analytical solution: sin(x)sin(y)exp(-2vt)
        return u_true(x,t, self.params[0])

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
        def get_gamma(W, xs, tol=1e-4, **kwags):
            J = get_J(W, xs)
            N = get_N(W, xs)
            matvec = lambda x: jnp.dot(J.T @ J, x)
            gamma = jaxopt.linear_solve.solve_normal_cg(matvec, J.T @ N, tol=tol, **kwags)
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

    def ode(self, t,y, args):
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
    def __init__(self, nn):
        self.nn = nn

    def __call__(self, x):
        L = 1
        omega = jnp.ones_like(x) * 2 * jnp.pi / L
        embed_v = jnp.concatenate([jnp.sin(omega * x), jnp.cos(omega * x)], axis=-1)
        return self.nn(embed_v)

# Setup PDE 


# Plot comparison in 2D
# Plot comparison in section
def plot_sections(ax, y, sol, evon, pde, u_true, label=None):
    xs = jnp.linspace(*pde.xspan[0], 100)
    for w, t in zip(sol.ys[::10], sol.ts[::10]):

        nn = evon.new_w(w).get_nn()
        u_trueF = lambda x: pde.u_true(jnp.array([x]), t)
        u_predF = lambda x: jnp.sum(nn(jnp.array([x])))

        u_true = jax.vmap(u_trueF)(xs)
        u_pred = jax.vmap(u_predF)(xs)
        ax.plot(xs, u_true, linewidth=3, color="blue",)
        ax.plot(xs, u_pred, linewidth=3, color="red", linestyle="--")

    ax.plot(xs, u_true, linewidth=3, color="blue", label="Analytical Sol.")
    ax.plot(xs, u_pred, linewidth=3, color="red", linestyle="--", label="Prediction")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("u")
# Error versus time
def plot_error(ax, sol, evon, pde, u_true, label=None):
    X = jnp.linspace(*pde.xspan[0], 100).reshape(-1,1)
    u_true0 = jax.vmap(pde.u_true, in_axes=(0,None))(X.ravel(), 0)
    err = []
    u_trues = np.array([])
    u_preds = np.array([])
    for w,t in zip(sol.ys, sol.ts):
        nn = evon.new_w(w).get_nn()
        u_predf = lambda x: jnp.sum(nn(x))
        u_true = jax.vmap(pde.u_true, in_axes=(0,None))(X.ravel(), t)
        u_pred = jax.vmap(u_predf)(X)
        error = jnp.linalg.norm(u_true - u_pred) / jnp.linalg.norm(u_true0)
        err.append(error)
        u_trues = np.append(u_trues, u_true)
        u_preds = np.append(u_preds, u_pred)
    l2_err = jnp.sqrt(jnp.mean((u_trues.ravel() - u_preds.ravel())**2))
    print(err[-1])
    print(f"L2 error: {l2_err:.2e}")
    ax.plot(sol.ts, err, color="black", label=None)
    ax.set_xlabel("t")
    ax.set_ylabel("Relative Error")
    ax.set_yscale("log")
    return ax, err


#%%
if __name__ == "__main__":
    """
    Neural Network
    """
    key = jr.PRNGKey(0)
    pde = ParabolicPDE1D(jnp.array([0.05]), jnp.array([[0., 1]]), jnp.array([0., 1.]))


    # Learn initial condition
    opt = optax.adam(learning_rate=optax.exponential_decay(1e-3, 2000, 0.9, end_value=1e-4))
    nbatch = 5000
    nn = DrichletNN(eqx.nn.MLP(2, 1, 10, 4, activation=jnp.tanh,key=jax.random.PRNGKey(0)))
    #nn = eqx.nn.MLP(1, 1, 10, 4, activation=jnp.tanh,key=jax.random.PRNGKey(0))
    evonn = EvolutionalNN.from_nn(nn, pde)
    time_str = time()
    _evonnfit = evonn.fit_initial(nbatch, 10_000, opt, key)
    nn2 = _evonnfit.get_nn()
    evonnfit = EvolutionalNN.from_nn(nn2, pde)
    xspans = pde.xspan
    gen_xgrid = lambda xspan: jnp.linspace(xspan[0]+1e-4, xspan[1]-1e-4, 300)
    xs_grids = jax.vmap(gen_xgrid)(xspans)
    Xg = jnp.meshgrid(*xs_grids)
    xs = jnp.stack([Xg[i].ravel() for i in range(len(Xg))]).T
    #evonnfit.get_N(evonnfit.W, xs)
    #evonnfit.get_J(evonnfit.W, xs)
    #g = evonnfit.get_gamma(evonnfit.W, xs)
    #print(g)

    # Evolve
    term = dfx.ODETerm(evonnfit.ode)
    solver = dfx.Euler()
    #stepsize_controller = dfx.PIDController(rtol=1e-4, atol=1e-4)
    t1 = 1
    stepsize_controller = dfx.ConstantStepSize()
    saveat = dfx.SaveAt(ts=np.linspace(pde.tspan[0], t1, 100).tolist())
    str2_time = time()
    sol = dfx.diffeqsolve(term, solver, t0=pde.tspan[0], t1=t1, dt0=0.001, y0=evonnfit.W, saveat=saveat, stepsize_controller=stepsize_controller, progress_meter=dfx.TqdmProgressMeter(refresh_steps=2))
    end_time = time()
    print("Time elapsed: ", end_time - time_str)
    print("Time elapsed for evolution: ", end_time - str2_time)

    fig, ax = plt.subplots()
    plot_sections(ax, 1., sol, evonnfit, pde, pde.u_true)
    fig5, ax5 = plt.subplots()
    _, err_nn = plot_error(ax5, sol, evonnfit, pde, pde.u_true)
    print("Number of parameters: ", evonnfit.W.shape[0])


# %%
