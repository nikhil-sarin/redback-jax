"""Angular discretisation and observer geometry."""

from functools import partial

import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnames=("resolution",))
def angular_mesh(theta_jet, resolution=50):
    """Return solid angles and cell-centre coordinates for a polar jet mesh."""
    latitude_step = theta_jet / resolution
    azimuth_step = 2.0 * jnp.pi / resolution
    theta_edges = jnp.arange(resolution + 1) * latitude_step
    solid_angle_latitude = azimuth_step * (
        jnp.cos(theta_edges[:-1]) - jnp.cos(theta_edges[1:])
    )
    solid_angle = jnp.repeat(solid_angle_latitude, resolution)
    theta = (jnp.arange(resolution) + 0.5) * latitude_step
    phi = (jnp.arange(resolution) + 0.5) * azimuth_step
    return solid_angle, theta, phi


@jit
def angular_patch_coordinates(theta, phi):
    """Expand one-dimensional polar and azimuthal grids into flat patches."""
    return jnp.repeat(theta, phi.size), jnp.tile(phi, theta.size)


@jit
def observer_angle(phi, theta, theta_observer, phi_observer=0.0):
    """Return the angle between each angular cell and the line of sight."""
    cosine = jnp.cos(theta_observer) * jnp.cos(theta)[:, None] + jnp.sin(
        theta_observer
    ) * jnp.sin(theta)[:, None] * jnp.cos(phi[None, :] - phi_observer)
    return jnp.arccos(jnp.clip(cosine, -1.0, 1.0)).reshape(-1)


@jit
def observer_angle_patches(phi, theta, theta_observer, phi_observer=0.0):
    """Return viewing angles for already flattened angular patches."""
    cosine = jnp.cos(theta_observer) * jnp.cos(theta) + jnp.sin(
        theta_observer
    ) * jnp.sin(theta) * jnp.cos(phi - phi_observer)
    return jnp.arccos(jnp.clip(cosine, -1.0, 1.0))
