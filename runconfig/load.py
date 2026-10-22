"""Config files -> validated, fully resolved Config objects, and back (config.resolved.yaml)."""
import dataclasses
import types
import typing
import warnings
from pathlib import Path
from typing import Any

import yaml

from . import legacy
from .schema import ENCODERS, METHODS, SCHEDULES, SECTIONS, VERSION, Config, ConfigError

__all__ = ["load_config", "config_from_dict", "save_resolved"]

_COMMON_MSFA_KEYS = frozenset({"method", "n_bands", "encoder"})
_ADAM_EPS = 1e-8


def load_config(path: str | Path) -> Config:
    """Read a config file, old or new, and return it validated and resolved."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ConfigError(f"{path}: empty config file")
    return config_from_dict(raw, source=str(path))


def config_from_dict(raw: Any, source: str = "<dict>") -> Config:
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: expected a mapping at the top level, got {type(raw).__name__}")
    if "version" not in raw:
        raw = legacy.translate(raw, source)
    elif raw["version"] != VERSION:
        raise ConfigError(f"{source}: version {raw['version']!r}; this code reads version {VERSION}")
    unknown = sorted(set(raw) - {"version"} - set(SECTIONS))
    if unknown:
        raise ConfigError(f"{source}: unknown section(s) {unknown}; allowed: {sorted(SECTIONS)}")
    sections = {name: _section(cls, raw.get(name), f"{source}: {name}")
                for name, cls in SECTIONS.items()}
    cfg = _resolve(Config(version=VERSION, **sections), source)
    _check(cfg, source)
    return cfg


def save_resolved(cfg: Config, run_dir: str | Path) -> Path:
    """Write config.resolved.yaml: every key, every default filled in. Loading it gives `cfg` back."""
    path = Path(run_dir) / "config.resolved.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(_plain(dataclasses.asdict(cfg)), f, sort_keys=False)
    return path


# ---------------------------------------------------------------- one section

def _section(cls: type, values: Any, where: str):
    if values is None:                               # absent, or a header with nothing under it
        values = {}
    if not isinstance(values, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(values).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - set(fields))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(fields)}")
    missing = sorted(n for n, f in fields.items() if n not in values
                     and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING)
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {missing}")
    return cls(**{k: _coerce(v, hints[k], f"{where}.{k}") for k, v in values.items()})


def _coerce(value: Any, tp: Any, where: str) -> Any:
    """Check `value` against the annotation `tp`. Ints are accepted for floats, lists for tuples."""
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        if len(inner) != 1:
            raise TypeError(f"{where}: unsupported annotation {tp}")      # a schema bug, not a config error
        return _coerce(value, inner[0], where)
    if origin is tuple:                                                    # tuple[int, ...]
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        return tuple(_coerce(v, args[0], f"{where}[{i}]") for i, v in enumerate(value))
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true or false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            hint = " (YAML reads a number like 2e-4 as text; write 2.0e-4)" if isinstance(value, str) else ""
            raise ConfigError(f"{where}: expected a number, got {value!r}{hint}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise TypeError(f"{where}: unsupported annotation {tp}")


# ---------------------------------------------------------------- method defaults

def _resolve(cfg: Config, source: str) -> Config:
    m = cfg.msfa
    spec = METHODS.get(m.method)
    if spec is None:
        raise ConfigError(f"{source}: unknown msfa.method {m.method!r}; known: {sorted(METHODS)}")
    stray = sorted(f.name for f in dataclasses.fields(m)
                   if getattr(m, f.name) is not None and f.name not in _COMMON_MSFA_KEYS | spec.keys)
    if stray:
        raise ConfigError(f"{source}: msfa key(s) {stray} don't apply to method {m.method!r}")
    missing = sorted(k for k in spec.required if getattr(m, k) is None)
    if missing:
        raise ConfigError(f"{source}: method {m.method!r} needs msfa key(s) {missing}")

    filled = {k: v for k, v in spec.defaults.items() if getattr(m, k) is None and k != "init_std"}
    m = dataclasses.replace(m, **filled)
    for key, options in spec.choices.items():
        if getattr(m, key) not in options:
            raise ConfigError(f"{source}: msfa.{key} {getattr(m, key)!r} not available for method "
                              f"{m.method!r}; choose from {list(options)}")
    if "init_std" in spec.keys:                      # only meaningful for init "normal"
        if m.init == "normal" and m.init_std is None:
            m = dataclasses.replace(m, init_std=spec.defaults["init_std"])
        elif m.init != "normal" and m.init_std is not None:
            raise ConfigError(f"{source}: msfa.init_std only applies to init 'normal', not {m.init!r}")
    if m.encoder not in ENCODERS:
        raise ConfigError(f"{source}: unknown msfa.encoder {m.encoder!r}; choose from {list(ENCODERS)}")
    if m.schedule is not None and m.schedule not in SCHEDULES:
        raise ConfigError(f"{source}: unknown msfa.schedule {m.schedule!r}; choose from {list(SCHEDULES)}")

    t = cfg.train
    if spec.learned:
        t = dataclasses.replace(t, pattern_lr=t.lr if t.pattern_lr is None else t.pattern_lr,
                                pattern_eps=_ADAM_EPS if t.pattern_eps is None else t.pattern_eps)
    elif t.pattern_lr is not None or t.pattern_eps is not None:
        raise ConfigError(f"{source}: train.pattern_lr and train.pattern_eps don't apply to the "
                          f"fixed method {m.method!r}")
    return dataclasses.replace(cfg, msfa=m, train=t)


# ---------------------------------------------------------------- cross-field checks

def _check(cfg: Config, source: str) -> None:
    m, p, t, f = cfg.msfa, cfg.preprocess, cfg.train, cfg.finetune
    learned = METHODS[m.method].learned

    def fail(msg: str) -> None:
        raise ConfigError(f"{source}: {msg}")

    for name in ("epochs", "batch_train", "batch_test", "checkpoint_interval"):
        if getattr(t, name) < 1:
            fail(f"train.{name} must be at least 1")
    for name, value in dataclasses.asdict(cfg.seeds).items():
        if value is not None and value < 0:
            fail(f"seeds.{name} must be non-negative")
    if m.n_bands < 2:
        fail("msfa.n_bands must be at least 2")
    if p.illuminant not in ("d65", "none"):
        fail(f"unknown preprocess.illuminant {p.illuminant!r}")
    if p.observer == "tokyotech":
        if m.n_bands != 5:
            fail(f"observer 'tokyotech' gives 5 bands, msfa.n_bands is {m.n_bands}")
    elif p.observer in ("ideal", "gaussian"):
        if len(p.wls_out) != m.n_bands:
            fail(f"preprocess.wls_out has {len(p.wls_out)} peaks, msfa.n_bands is {m.n_bands}")
    else:
        fail(f"unknown preprocess.observer {p.observer!r}")
    if m.method == "tokyotech_rgbco" and m.n_bands != 5:
        fail("method 'tokyotech_rgbco' is a 5-band pattern")

    if f.epoch_freeze is not None:
        if not 1 <= f.epoch_freeze < t.epochs:
            fail(f"finetune.epoch_freeze {f.epoch_freeze} must be in 1..{t.epochs - 1}")
        if not learned:
            warnings.warn(f"{source}: epoch_freeze with the fixed method {m.method!r}: the old train.py "
                          "then trained two phases with the same pattern (results in finetune/)",
                          stacklevel=3)
    elif learned:
        warnings.warn(f"{source}: learned method {m.method!r} without finetune.epoch_freeze: "
                      "the run ends with a soft pattern", stacklevel=3)


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x
