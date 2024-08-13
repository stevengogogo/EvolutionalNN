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
        def method_opt(A,b, initial_gamma=0):
            loss = lambda x: jnp.mean(jnp.square(A @ x - b))
            x = jaxopt.ScipyMinimize(loss, x0=initial_gamma, **kwargs)
            return x
        
        if strategy == "inverse":
            method = method_inverse
        elif strategy == "gradient_descent":
            method = method_opt
        else: 
            raise NotImplementedError(f"Strategy {strategy} not implemented")

        self.method = method

solvers = [LinearSolver(strategy="inverse"), 
           LinearSolver(strategy="gradient_descent", method="L-BFGS-B")]


