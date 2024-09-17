""" 
Solve Linear System with Gradient Descent
"""
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
import numpy as np


class LinearSolver(eqx.Module):
    """Solve A @ x = b

    Args:
        eqx (_type_): _description_
    """
    method: Callable

    def __init__(self, strategy="inverse", **kwargs):

        @jax.jit
        def method_inverse(A, b):
            return jnp.linalg.solve(A, b, **kwargs)

        @jax.jit
        def method_opt(A,b, **kwags):
            def matvec_A(x):
                return  jnp.dot(A, x)
            return jaxopt.linear_solve.solve_normal_cg(A, b, **kwags)
        
        if strategy == "inverse":
            method = method_inverse
        elif strategy == "gradient_descent":
            method = method_opt
        else: 
            raise NotImplementedError(f"Strategy {strategy} not implemented")

        self.method = method
    
    def __call__(self, A, b, **kwargs):
        return self.method(A, b, **kwargs)

solvers = [LinearSolver(strategy="inverse"), 
           LinearSolver(strategy="gradient_descent", method="L-BFGS-B")]


a = jr.normal(jr.PRNGKey(2), shape=(2, 5))
b = jr.normal(jr.PRNGKey(1), shape=(2,))

x1 = solvers[0](a.T @ a, a.T @ b)
x2 = solvers[1](a.T @ a, a.T @ b)

assert jnp.allclose(a.T @ a @x1, a.T @ b)
# %%
A = jr.normal(jr.PRNGKey(2), shape=(2, 5))
b = jr.normal(jr.PRNGKey(1), shape=(2,))

def matvec_A(x):
  return  jnp.dot(A.T @ A, x)

sol = jaxopt.linear_solve.solve_normal_cg(matvec_A, A.T @ b, tol=1e-5)
print(sol)

assert jnp.allclose(A.T @ A  @ sol, A.T @ b)
# %%
