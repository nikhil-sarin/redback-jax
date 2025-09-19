"""
Basic tests for redback_jax package.
"""
import pytest
import redback_jax


def test_package_import():
    """Test that the package can be imported."""
    assert redback_jax is not None


def test_package_version():
    """Test that package version is accessible."""
    assert hasattr(redback_jax, '__version__')
    assert isinstance(redback_jax.__version__, str)
    assert redback_jax.__version__ == "0.1.0"


def test_package_author():
    """Test that package author is accessible."""
    assert hasattr(redback_jax, '__author__')
    assert redback_jax.__author__ == "Nikhil Sarin"


def test_submodules_importable():
    """Test that submodules can be imported."""
    from redback_jax import models, utils, inference
    
    assert models is not None
    assert utils is not None
    assert inference is not None