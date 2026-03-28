"""
Tests for the redback_jax model registry and plugin system.
"""
import warnings
import pytest

from redback_jax.models.registry import MODEL_REGISTRY, register_model, get_model, load_plugins


# ---------------------------------------------------------------------------
# register_model / get_model
# ---------------------------------------------------------------------------

def test_register_and_get_model():
    """register_model stores a callable that get_model retrieves."""
    def dummy_model():
        return 42

    register_model("_test_dummy", dummy_model)
    assert get_model("_test_dummy") is dummy_model


def test_get_model_unknown_raises():
    """get_model raises KeyError with a helpful message for unknown names."""
    with pytest.raises(KeyError, match="not found in redback_jax registry"):
        get_model("_definitely_not_registered_xyz")


def test_get_model_error_lists_available():
    """The KeyError message lists available models."""
    register_model("_listed_model", lambda: None)
    with pytest.raises(KeyError, match="_listed_model"):
        get_model("_not_this_one_xyz")


def test_register_model_overwrites():
    """Re-registering a name silently replaces the previous entry."""
    register_model("_overwrite_test", lambda: 1)
    new_fn = lambda: 2
    register_model("_overwrite_test", new_fn)
    assert get_model("_overwrite_test") is new_fn


def test_register_model_any_callable():
    """register_model accepts any callable — class, lambda, or function."""
    class ModelClass:
        pass

    register_model("_class_model", ModelClass)
    assert get_model("_class_model") is ModelClass

    register_model("_lambda_model", lambda x: x)
    assert callable(get_model("_lambda_model"))


# ---------------------------------------------------------------------------
# MODEL_REGISTRY dict
# ---------------------------------------------------------------------------

def test_model_registry_is_dict():
    assert isinstance(MODEL_REGISTRY, dict)


def test_builtin_models_registered():
    """Built-in arnett models are registered when the models package is imported."""
    # Import the package so __init__.py runs register_model for builtins
    import redback_jax.models  # noqa: F401
    assert "arnett_bolometric" in MODEL_REGISTRY
    assert "arnett_with_features_cosmology" in MODEL_REGISTRY


def test_get_model_returns_builtin_callable():
    import redback_jax.models  # noqa: F401
    from redback_jax.models.supernova_models import arnett_bolometric
    assert get_model("arnett_bolometric") is arnett_bolometric


# ---------------------------------------------------------------------------
# load_plugins — entry-point discovery
# ---------------------------------------------------------------------------

def test_load_plugins_no_installed_plugins():
    """load_plugins runs without error when no plugins are installed."""
    load_plugins()  # should not raise


def test_load_plugins_bad_register_fn_warns(monkeypatch):
    """A plugin whose register function raises issues an ImportWarning."""
    from importlib.metadata import EntryPoint

    bad_ep = EntryPoint(name="bad_plugin", group="redback_jax.models",
                        value="tests.unit.models.test_registry:_bad_register")

    def _mock_entry_points(group):
        if group == "redback_jax.models":
            return [bad_ep]
        return []

    monkeypatch.setattr("redback_jax.models.registry.entry_points", _mock_entry_points)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_plugins()

    import_warnings = [w for w in caught if issubclass(w.category, ImportWarning)]
    assert len(import_warnings) == 1
    assert "bad_plugin" in str(import_warnings[0].message)


def _bad_register(registry):
    raise RuntimeError("intentional failure")


def test_load_plugins_calls_register_fn(monkeypatch):
    """load_plugins calls register(registry) for each discovered entry-point."""
    from importlib.metadata import EntryPoint

    called_with = []

    def good_register(registry):
        called_with.append(registry)
        registry["_plugin_model"] = lambda: "from_plugin"

    good_ep = EntryPoint(name="good_plugin", group="redback_jax.models",
                         value="tests.unit.models.test_registry:_good_register_stub")

    # Patch ep.load() to return our function directly
    good_ep_mock = type("EP", (), {
        "name": "good_plugin",
        "load": lambda self: good_register,
    })()

    def _mock_entry_points(group):
        if group == "redback_jax.models":
            return [good_ep_mock]
        return []

    monkeypatch.setattr("redback_jax.models.registry.entry_points", _mock_entry_points)

    load_plugins()

    assert len(called_with) == 1
    assert called_with[0] is MODEL_REGISTRY
    assert MODEL_REGISTRY.get("_plugin_model") is not None


def _good_register_stub(registry):
    pass  # placeholder; actual fn injected via monkeypatch above


def test_load_plugins_bad_plugin_does_not_affect_others(monkeypatch):
    """A failing plugin does not prevent subsequent plugins from loading."""
    registered = []

    def bad_register(registry):
        raise ValueError("boom")

    def good_register(registry):
        registered.append("ok")

    eps = [
        type("EP", (), {"name": "bad", "load": lambda self: bad_register})(),
        type("EP", (), {"name": "good", "load": lambda self: good_register})(),
    ]

    monkeypatch.setattr(
        "redback_jax.models.registry.entry_points",
        lambda group: eps if group == "redback_jax.models" else [],
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        load_plugins()

    assert registered == ["ok"]


# ---------------------------------------------------------------------------
# Public API surface via redback_jax.models
# ---------------------------------------------------------------------------

def test_registry_symbols_exported():
    """The registry symbols are accessible from the top-level models package."""
    import redback_jax.models as m
    assert hasattr(m, "MODEL_REGISTRY")
    assert hasattr(m, "register_model")
    assert hasattr(m, "get_model")
    assert hasattr(m, "load_plugins")


def test_registry_in_all():
    import redback_jax.models as m
    for sym in ("MODEL_REGISTRY", "register_model", "get_model", "load_plugins"):
        assert sym in m.__all__
