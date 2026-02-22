"""Connector factory and registration."""

from __future__ import annotations

from backend.integrations.base import BaseConnector, ConnectorConfig

_REGISTRY: dict[str, type[BaseConnector]] = {}


def register_connector(platform: str, cls: type[BaseConnector]) -> None:
    """Register a connector class for a platform."""
    _REGISTRY[platform] = cls


def get_connector(config: ConnectorConfig) -> BaseConnector:
    """Instantiate the appropriate connector for the given config.

    Raises ValueError if the platform is not registered.
    """
    cls = _REGISTRY.get(config.platform)
    if cls is None:
        raise ValueError(
            f"Unknown platform '{config.platform}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return cls(config)


def list_platforms() -> list[str]:
    """Return all registered platform names."""
    return sorted(_REGISTRY.keys())


def _register_all() -> None:
    """Import all connector modules to trigger registration."""
    from backend.integrations import google_ads, meta_ads, ga4, shopify, woocommerce  # noqa: F401


_register_all()
