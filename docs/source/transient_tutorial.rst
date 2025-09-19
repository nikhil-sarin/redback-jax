Transient Analysis Tutorial
==========================

This tutorial demonstrates how to use the JAX-friendly :class:`redback_jax.Transient` and :class:`redback_jax.Spectrum` classes for electromagnetic transient analysis.

Basic Transient Usage
---------------------

Creating a Transient Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most basic way to create a transient is to provide time and flux data:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from redback_jax import Transient
   
   # Generate synthetic lightcurve data
   time = np.linspace(0, 20, 30)
   flux = 5 * np.exp(-time/8) + 0.2 * np.random.randn(30)
   flux_err = 0.1 * np.abs(flux) + 0.05
   
   # Create transient object
   transient = Transient(
       time=time,
       y=flux,
       y_err=flux_err,
       data_mode='flux',
       name='My Transient',
       redshift=0.05
   )

The :class:`Transient` class automatically converts input arrays to JAX arrays for efficient computation.

Data Modes
~~~~~~~~~~

The transient class supports several data modes:

- ``'flux'``: Flux measurements (erg cm⁻² s⁻¹)
- ``'magnitude'``: Magnitude measurements
- ``'luminosity'``: Luminosity measurements (erg s⁻¹)
- ``'flux_density'``: Flux density measurements (mJy)
- ``'counts'``: Count measurements

.. code-block:: python

   # Magnitude data
   magnitude_transient = Transient(
       time=time,
       y=magnitude,
       y_err=magnitude_err,
       data_mode='magnitude'
   )

Loading Data from Files
~~~~~~~~~~~~~~~~~~~~~~~

You can load transient data directly from CSV or text files:

.. code-block:: python

   # Load from CSV file
   transient = Transient.from_data_file(
       'lightcurve_data.csv',
       data_mode='flux',
       name='SN2023A'
   )
   
   # Specify custom column names
   transient = Transient.from_data_file(
       'data.txt',
       data_mode='magnitude',
       time_col='mjd',
       y_col='mag',
       y_err_col='mag_err'
   )

Plotting Data
~~~~~~~~~~~~~

The transient class provides simple plotting functionality:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Basic plot
   fig, ax = plt.subplots(figsize=(10, 6))
   transient.plot_data(axes=ax, color='blue', alpha=0.7)
   plt.show()
   
   # Plot without error bars
   transient.plot_data(show_errors=False, color='red')

Multi-band Photometry
---------------------

For multi-band photometric data, include band information:

.. code-block:: python

   # Multi-band data
   time = np.repeat(np.linspace(0, 10, 10), 3)  # 3 bands × 10 epochs
   bands = np.tile(['g', 'r', 'i'], 10)
   magnitude = np.random.uniform(16, 20, 30)  # Random magnitudes
   
   transient = Transient(
       time=time,
       y=magnitude,
       data_mode='magnitude',
       bands=bands,
       name='Multi-band SN'
   )
   
   # Plot different bands with different colors
   colors = {'g': 'green', 'r': 'red', 'i': 'orange'}
   fig, ax = plt.subplots(figsize=(10, 6))
   
   for band in ['g', 'r', 'i']:
       mask = transient.bands == band
       time_band = transient.time[mask]
       mag_band = transient.y[mask]
       
       ax.scatter(np.asarray(time_band), np.asarray(mag_band), 
                 color=colors[band], label=f'{band} band')
   
   ax.set_xlabel(transient.xlabel)
   ax.set_ylabel(transient.ylabel)
   ax.invert_yaxis()  # Magnitudes increase downward
   ax.legend()

Model Fitting and Plotting
--------------------------

You can overplot models on your data:

.. code-block:: python

   # Define a simple exponential model
   def exponential_model(t, amplitude, decay_time, offset):
       return amplitude * jnp.exp(-t / decay_time) + offset
   
   # Model parameters
   model_params = {
       'amplitude': 3.0,
       'decay_time': 8.0, 
       'offset': 0.5
   }
   
   # Plot data with model
   fig, ax = plt.subplots(figsize=(10, 6))
   transient.plot_data(axes=ax, label='Data')
   transient.plot_model(
       model_func=exponential_model,
       model_params=model_params,
       axes=ax,
       color='red',
       label='Model'
   )

Working with JAX Arrays
-----------------------

All data in the transient object are stored as JAX arrays, enabling efficient computation:

.. code-block:: python

   # Access data as JAX arrays
   peak_time = jnp.argmax(transient.y)
   peak_flux = jnp.max(transient.y)
   mean_time = jnp.mean(transient.time)
   
   # JAX operations work directly
   log_flux = jnp.log10(transient.y)
   flux_gradient = jnp.gradient(transient.y)
   
   # Use JAX transformations
   import jax
   
   def compute_chi_squared(model_params):
       model_flux = exponential_model(transient.time, **model_params)
       residuals = (transient.y - model_flux) / transient.y_err
       return jnp.sum(residuals**2)
   
   # Get gradients with respect to parameters
   chi2_grad = jax.grad(compute_chi_squared)

Spectrum Analysis
----------------

The :class:`Spectrum` class handles spectroscopic data:

Creating a Spectrum
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from redback_jax import Spectrum
   
   # Create spectrum data
   wavelength = np.linspace(4000, 7000, 200)  # Angstroms
   flux_density = np.random.random(200)  # Flux density
   flux_err = 0.1 * flux_density
   
   spectrum = Spectrum(
       wavelength=wavelength,
       flux_density=flux_density,
       flux_density_err=flux_err,
       name='SN Spectrum',
       time='Day 5'
   )

Plotting Spectra
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot spectrum
   fig, ax = plt.subplots(figsize=(12, 6))
   spectrum.plot_data(axes=ax, color='purple', alpha=0.8)
   plt.show()

Working with JAX in Spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Spectrum operations with JAX
   peak_wavelength = spectrum.wavelength[jnp.argmax(spectrum.flux_density)]
   total_flux = jnp.trapz(spectrum.flux_density, spectrum.wavelength)
   
   # Smooth spectrum with JAX
   from scipy import ndimage
   smoothed_flux = ndimage.gaussian_filter1d(
       np.asarray(spectrum.flux_density), sigma=2
   )

Advanced Usage
--------------

Performance Tips
~~~~~~~~~~~~~~~

1. **Use JAX arrays from the start**: If your data is already in JAX format, pass it directly to avoid unnecessary conversions.

2. **Batch operations**: Process multiple transients using JAX's vectorization capabilities.

3. **JIT compilation**: Use ``jax.jit`` to compile model functions for faster evaluation.

.. code-block:: python

   import jax
   
   # JIT-compile model for speed
   @jax.jit
   def fast_exponential_model(t, amplitude, decay_time, offset):
       return amplitude * jnp.exp(-t / decay_time) + offset

Custom Data Processing
~~~~~~~~~~~~~~~~~~~~~

Since all data are JAX arrays, you can easily implement custom processing:

.. code-block:: python

   def calculate_color(transient_g, transient_r):
       """Calculate g-r color from two transients."""
       # Interpolate r-band to g-band times
       r_interp = jnp.interp(transient_g.time, transient_r.time, transient_r.y)
       color = transient_g.y - r_interp
       return color
   
   def phase_fold(transient, period, epoch):
       """Phase-fold a transient lightcurve."""
       phase = jnp.mod(transient.time - epoch, period) / period
       return phase

Error Handling
--------------

The transient classes include validation to catch common errors:

.. code-block:: python

   # This will raise a ValueError
   try:
       bad_transient = Transient(
           time=np.array([1, 2, 3]),
           y=np.array([1, 2]),  # Wrong length!
           data_mode='flux'
       )
   except ValueError as e:
       print(f"Error: {e}")
   
   # This will also raise a ValueError
   try:
       bad_mode = Transient(
           time=np.array([1, 2, 3]),
           y=np.array([1, 2, 3]),
           data_mode='invalid_mode'  # Invalid data mode!
       )
   except ValueError as e:
       print(f"Error: {e}")

Next Steps
----------

- Explore the :doc:`api` documentation for complete method references
- Check out the example scripts in the ``examples/`` directory
- Learn about JAX-based model fitting in the models tutorial