"""
Incompressible Navier-Stokes equation: Taylor-Green vortex
Ref: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.045303
"""
#%%
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import diffrax as dfx
from time import time
import numpy as np
import jax
from parabolic1d_drichlet import PDE, EvolutionalNN  

def u_true(x,y, t, nu, U0):
    u = U0 * jnp.cos(x) * jnp.sin(y) * jnp.exp(-2*nu*t)
    v = -U0*jnp.sin(x)*jnp.cos(y)*jnp.exp(-2*nu*t)
    return jnp.array([u, v])

class TaylorGreenVortex2D(PDE):
    def get_params(self):
        v = self.params[0]
        U0 = self.params[1]
        return v, U0
    
    def init_func(self, x):
        nu, U0 = self.get_params()
        return u_true(x[0], x[1], 0., nu, U0)

    def boundary_func(self, x):
        return jnp.array([0., 0.])
    
    def spatial_diff_operator(self, func):
        # func: U(x,y) -> u, v

        
        u_func = lambda x, y: jnp.sum(func(jnp.array([x, y]))[0])
        v_func = lambda x, y: jnp.sum(func(jnp.array([x, y]))[1])

        ux_func = jax.grad(u_func, argnums=0)
        uxx_func = jax.grad(ux_func, argnums=0)
        uy_func = jax.grad(u_func, argnums=1)
        uyy_func = jax.grad(uy_func, argnums=1)
        vx_func = jax.grad(v_func, argnums=0)
        vxx_func = jax.grad(vx_func, argnums=0)
        vy_func = jax.grad(v_func, argnums=1)
        vyy_func = jax.grad(vy_func, argnums=1)

        def Nx_func(X):
            nu, _ = self.get_params()
            x, y = X
            
            u = u_func(x, y)
            v = v_func(x, y)
            ux = ux_func(x, y)
            uxx = uxx_func(x, y)
            uy = uy_func(x, y)
            uyy = uyy_func(x, y)
            vx = vx_func(x, y)
            vxx = vxx_func(x, y)
            vy = vy_func(x, y)
            vyy = vyy_func(x, y)

            dudt = - u * ux - v * uy + nu * (uxx + uyy)
            dvdt = -u*ux - v * vy + nu * (vxx + vyy)
            return jnp.array([dudt, dvdt])
        
        return Nx_func

class DivergenceFreeNN2D(eqx.Module):
    nn: eqx.Module # input: 2 -> output: 1

    def __init__(self, width_size, depth, key, **kwargs):
        self.nn = eqx.nn.MLP(in_size=2, out_size=1, width_size=width_size, depth=depth, key=key, activation=jax.nn.tanh, **kwargs)
    
    def __call__(self, x):
        func = lambda x,y: jnp.sum(self.nn(jnp.array([x, y])))
        func_x = jax.grad(func, argnums=0)
        func_y = jax.grad(func, argnums=1)

        u_f = lambda x,y: func_y(x,y)
        v_f = lambda x,y: -func_x(x,y)

        u = u_f(x[0], x[1])
        v = v_f(x[0], x[1])
        return jnp.array([u, v])


key = jr.PRNGKey(0)
nbatch = 1000
nstep = 10000
pde = TaylorGreenVortex2D(params=jnp.array([1, 1.]), 
                          xspan=jnp.array([[0., 2*jnp.pi], [0, 2*jnp.pi]]), 
                          tspan=jnp.array([0., 1.]))

opt = optax.adam(learning_rate=optax.exponential_decay(1e-4, 1000, 0.9))

nn = DivergenceFreeNN2D(width_size=30, depth=4, key=key)
#nn = eqx.nn.MLP(2, 2, 10, 4, activation=jnp.tanh,key=jax.random.PRNGKey(0))
evonn = EvolutionalNN.from_nn(nn, pde)
evonnfit = evonn.fit_initial(nbatch, nstep, opt, key)
# %%

xspans = pde.xspan
gen_xgrid = lambda xspan: jnp.linspace(xspan[0]+1e-4, xspan[1]-1e-4, 33)
xs_grids = jax.vmap(gen_xgrid)(xspans)
Xg = jnp.meshgrid(*xs_grids)
xs = jnp.stack([Xg[i].ravel() for i in range(len(Xg))]).T
#%%
N = evonnfit.get_N(evonnfit.W, xs)

#%%
J = evonnfit.get_J(evonnfit.W, xs)
#%%
g = evonnfit.get_gamma(evonnfit.W, xs)
#print(g)
# %%
# Evolve
term = dfx.ODETerm(evonnfit.get_ode(xs))
solver = dfx.Euler()
#stepsize_controller = dfx.PIDController(rtol=1e-4, atol=1e-4)
t1 = 1
stepsize_controller = dfx.ConstantStepSize()
saveat = dfx.SaveAt(ts=np.linspace(pde.tspan[0], t1, 100).tolist())
str2_time = time()
sol = dfx.diffeqsolve(term, solver, t0=pde.tspan[0], t1=t1, dt0=0.001, y0=evonnfit.W, saveat=saveat, stepsize_controller=stepsize_controller, progress_meter=dfx.TqdmProgressMeter(refresh_steps=2))
end_time = time()
#print("Time elapsed: ", end_time - time_str)
#print("Time elapsed for evolution: ", end_time - str2_time)
#%%
import matplotlib.pyplot as plt
@eqx.filter_jit
def loop2d(arr1, arr2, fun):
    funcex = jax.jit(lambda x,y: fun(jnp.stack([x,y])))
    fj = jax.vmap(funcex, in_axes=(0,0))
    fi = jax.vmap(fj, in_axes=(0,0))
    return fi(arr1, arr2)
w = 3
Y, X = np.mgrid[-w:w:100j, -w:w:100j]
UV = u_true(X, Y, 0, 1., 1.)
U = UV[0]
V = UV[1]
speed = np.sqrt(U**2 + V**2)
lw = 5*speed / speed.max()

fig, axs = plt.subplots()
#  Varying density along a streamline
axs.streamplot(X, Y, U, V, density=[0.5, 1], cmap='autumn', linewidth=lw)
axs.set_title('Varying Density')
# %%
i = 0
t = sol.ts[i]
nn = evonnfit.new_w(sol.ys[i,:]).get_nn()

Z =  loop2d(X, Y, nn)

U_nn = Z[:,:,0]
V_nn = Z[:,:,1]

speed = np.sqrt(U_nn**2 + V_nn**2)
lw = 5*speed / speed.max()

fig, axs = plt.subplots()
#  Varying density along a streamline
axs.streamplot(X, Y, U_nn, V_nn, density=[0.5, 1], cmap='autumn', linewidth=lw)
axs.set_title('Varying Density')
# %%
