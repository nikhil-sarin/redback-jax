Installation
============

Requirements
------------

redback-jax requires Python 3.8 or later.

Core dependencies:

* JAX (>= 0.4.0)
* NumPy (>= 1.20.0)
* SciPy (>= 1.7.0)
* Pandas (>= 1.3.0)
* Matplotlib (>= 3.5.0)
* Astropy (>= 4.0.0)

Installation from PyPI
----------------------

Once available on PyPI:

.. code-block:: bash

   pip install redback-jax

With optional dependencies:

.. code-block:: bash

   pip install redback-jax[all]

Installation from Source
------------------------

.. code-block:: bash

   git clone https://github.com/nikhil-sarin/redback-jax.git
   cd redback-jax
   pip install -e .

Development Installation
-----------------------

For development with all dependencies:

.. code-block:: bash

   git clone https://github.com/nikhil-sarin/redback-jax.git
   cd redback-jax
   pip install -e .[dev]

This installs testing, documentation, and development tools.