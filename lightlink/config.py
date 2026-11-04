"""YAML config loader with attribute access (cfg.demod.poll_interval_s)."""
from pathlib import Path
import copy
import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class Cfg:
    def __init__(self, data: dict):
        self._d = data

    def __getattr__(self, name):
        try:
            v = self._d[name]
        except KeyError:
            raise AttributeError(name) from None
        return Cfg(v) if isinstance(v, dict) else v

    def get(self, name, default=None):
        return self._d.get(name, default)

    def to_dict(self) -> dict:
        return copy.deepcopy(self._d)


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None, overrides: dict | None = None) -> Cfg:
    with open(path or DEFAULT_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if overrides:
        data = _merge(data, overrides)
    rate = data["protocol"]["bit_rate_bps"]
    if not 2.0 <= rate <= 15.0:
        raise ValueError(f"protocol.bit_rate_bps must be in 2..15, got {rate}")
    return Cfg(data)
