"""Model registry and plugin loader for redback-jax.

External packages register models via the ``redback_jax.models`` entry-point
group in their ``pyproject.toml``::

    [project.entry-points."redback_jax.models"]
    snmix = "snmix.redback_plugin:register"

The callable must accept the registry dict and add entries to it::

    def register(registry):
        from snmix.redback_plugin import nickelmixing_bolometric, collapsar_bolometric
        registry["nickelmixing_bolometric"] = nickelmixing_bolometric
        registry["collapsar_bolometric"] = collapsar_bolometric
"""

from importlib.metadata import entry_points

# Central model registry: name -> callable
MODEL_REGISTRY: dict = {}


def register_model(name: str, fn):
    """Register a model function under *name*."""
    MODEL_REGISTRY[name] = fn


def get_model(name: str):
    """Retrieve a registered model by name, or raise KeyError."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found in redback_jax registry. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


def load_plugins():
    """Discover and load all redback_jax.models entry-point plugins."""
    eps = entry_points(group="redback_jax.models")
    for ep in eps:
        try:
            register_fn = ep.load()
            register_fn(MODEL_REGISTRY)
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(
                f"Failed to load redback_jax plugin '{ep.name}': {exc}",
                ImportWarning,
                stacklevel=2,
            )
