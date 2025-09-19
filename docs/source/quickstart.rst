Quick Start
===========

Basic Usage
-----------

.. code-block:: python

   import redback_jax
   from redback_jax import Transient, Spectrum
   import numpy as np
   
   print(f"redback-jax version: {redback_jax.__version__}")

Creating Your First Transient
-----------------------------

.. code-block:: python

   # Generate some sample lightcurve data
   time = np.linspace(0, 10, 20)
   flux = 2 * np.exp(-time/5) + 0.1 * np.random.randn(20)
   flux_err = 0.1 * np.ones(20)
   
   # Create a transient object
   transient = Transient(
       time=time,
       y=flux,
       y_err=flux_err,
       data_mode='flux',
       name='My First Transient'
   )
   
   # Plot the data
   import matplotlib.pyplot as plt
   fig, ax = plt.subplots()
   transient.plot_data(axes=ax)
   plt.show()

Loading Data from Files
-----------------------

.. code-block:: python

   # Load transient data from a CSV file
   transient = Transient.from_data_file(
       'lightcurve.csv',
       data_mode='magnitude',
       name='SN2023A'
   )
   
   # Plot the loaded data
   transient.plot_data(color='red', show_errors=True)

Working with Spectra
--------------------

.. code-block:: python

   # Create a spectrum
   wavelength = np.linspace(4000, 7000, 100)
   flux_density = np.random.random(100)
   
   spectrum = Spectrum(
       wavelength=wavelength,
       flux_density=flux_density,
       name='Example Spectrum'
   )
   
   # Plot the spectrum
   spectrum.plot_data(color='purple')

Next Steps
----------

* Check out the :doc:`api` documentation for detailed function references
* Explore :doc:`examples` for common use cases
* See :doc:`contributing` for development guidelines