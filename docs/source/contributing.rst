Contributing
============

We welcome contributions to redback-jax! This guide will help you get started.

Development Setup
----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/your-username/redback-jax.git
   cd redback-jax

3. Install in development mode:

.. code-block:: bash

   pip install -e .[dev]

4. Install pre-commit hooks:

.. code-block:: bash

   pre-commit install

Running Tests
-------------

Run the test suite:

.. code-block:: bash

   pytest

Run tests with coverage:

.. code-block:: bash

   pytest --cov=redback_jax

Code Style
----------

We use several tools to maintain code quality:

* **Black**: Code formatting
* **isort**: Import sorting
* **flake8**: Linting

Format code:

.. code-block:: bash

   black redback_jax tests
   isort redback_jax tests

Check code style:

.. code-block:: bash

   flake8 redback_jax tests

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs
   make html

The documentation will be available in ``docs/_build/html/index.html``.

Submitting Changes
-----------------

1. Create a new branch for your feature:

.. code-block:: bash

   git checkout -b feature-name

2. Make your changes and add tests
3. Ensure all tests pass and code style is correct
4. Commit your changes with a clear message
5. Push to your fork and submit a pull request

Guidelines
----------

* Write tests for new functionality
* Follow existing code style and conventions
* Update documentation as needed
* Keep commits focused and atomic
* Write clear commit messages