import tomllib
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def load(path=ROOT / "config.toml"):
    with open(path, "rb") as f:
        return _ns(tomllib.load(f))


def resolve(p):
    """Resolve a path from config.toml relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p