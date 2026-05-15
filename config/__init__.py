"""Configuration package — exposes Settings singleton and YAML component map."""

from config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
