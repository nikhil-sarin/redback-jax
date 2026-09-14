redback-jax documentation
=========================

Welcome to Redback-JAX, a JAX-native companion to the `Redback
<https://github.com/nikhil-sarin/redback>`_ electromagnetic-transient
modeling and analysis stack.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   afterglow
   transient_tutorial
   api
   examples
   contributing

About redback-jax
----------------

Redback-JAX provides selected Redback models and composable JAX-native tools
for rapid electromagnetic-transient analysis and Bayesian inference. It is
designed to coexist with Redback rather than replace it: Redback supplies the
broader modeling and analysis ecosystem, while Redback-JAX focuses on workflows
that benefit from JIT compilation, vectorization, accelerator execution, and
automatic differentiation.

Features
--------

* **Rapid model evaluation**: JIT-compiled and vectorized transient calculations
* **Composable physics**: Flexible afterglow structure, dynamics, media, and radiation components
* **Bayesian inference**: Integration with modern probabilistic programming libraries
* **Automatic differentiation**: Gradient-based optimization and sampling on supported paths
* **GPU/TPU support**: Leverage JAX's hardware acceleration capabilities

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
