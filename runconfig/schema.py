"""Run configuration: one dataclass per config section, and the table of `msfa.method` names."""

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["VERSION", "ENCODERS", "SCHEDULES", "ConfigError", "MethodSpec",
           "METHODS", "TrainConfig", "RevSCIConfig", "MSFAConfig", "PreprocessConfig",
           "FinetuneConfig", "SeedsConfig", "Config", "SECTIONS"]

VERSION = 2
ENCODERS = ("masked_cube", "band_sum")
SCHEDULES = ("constant", "legacy_ramp")          # legacy_ramp: coef * 0.995**-epoch (old train.py's coef_func)


class ConfigError(ValueError):
    """A config that doesn't fit the schema."""


# ---------------------------------------------------------------- methods

@dataclass(frozen=True)
class MethodSpec:
    """What an `msfa.method` name means.

    Args:
        learned: the pattern is trained (the generator has parameters).
        keys: msfa keys the method accepts besides `method`, `n_bands` and `encoder`.
        required: keys among `keys` that must be set.
        defaults: values for keys left as None, `encoder` included.
        choices: allowed values of string keys, where restricted.
    """
    learned: bool
    keys: frozenset[str] = frozenset()
    required: frozenset[str] = frozenset()
    defaults: Mapping[str, Any] = field(default_factory=dict)
    choices: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


_FIXED_METHOD = MethodSpec(learned=False, defaults={"encoder": "masked_cube"})
_LEARNED_KEYS = frozenset({"init", "coef", "schedule"})
# the old runs' pairing: soft pattern x masked cube, which has no bottleneck (UnsoundEncoding)
_LEGACY_DEFAULTS = {"init": "uniform", "encoder": "masked_cube", "coef": 0.0, "schedule": "constant"}

METHODS: dict[str, MethodSpec] = {
    # fixed patterns, from msfa/registry.py
    "tokyotech_rgbco": _FIXED_METHOD,
    "osp": _FIXED_METHOD,
    "rand": _FIXED_METHOD,                           # drawn with the `pattern` seed
    "other": MethodSpec(learned=False, keys=frozenset({"msfa_path"}),
                        required=frozenset({"msfa_path"}), defaults={"encoder": "masked_cube"}),
    # learned patterns
    "softmax": MethodSpec(
        learned=True, keys=_LEARNED_KEYS | {"init_std", "alpha_end"},
        defaults={"init": "normal", "init_std": 0.05, "encoder": "band_sum",
                  "coef": 0.0, "schedule": "constant"},
        choices={"init": ("normal", "uniform", "constant")}),
    "legacy_softmax": MethodSpec(                    # old `tunable` + `norm: softmax`
        learned=True, keys=_LEARNED_KEYS | {"alpha_end"}, defaults=_LEGACY_DEFAULTS,
        choices={"init": ("uniform", "constant")}),
    "legacy_l1": MethodSpec(                         # old `tunable`, `norm` absent or "l1"
        learned=True, keys=_LEARNED_KEYS, defaults=_LEGACY_DEFAULTS,
        choices={"init": ("uniform", "constant")}),
}


# ---------------------------------------------------------------- sections

@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    epochs: int = 300
    lr: float = 2e-4
    decay_factor: float = 0.995
    amp: bool = True
    batch_train: int = 1
    batch_test: int = 1
    workers_train: int = 3
    workers_test: int = 3
    checkpoint_interval: int = 25
    metrics_report_interval: int = 10
    pattern_lr: float | None = None                  # learned methods only; None: lr
    pattern_eps: float | None = None                 # learned methods only; None: Adam's default, 1e-8


@dataclass(frozen=True, kw_only=True)
class RevSCIConfig:
    n_blocks: int = 24
    n_split: int = 2


@dataclass(frozen=True, kw_only=True)
class MSFAConfig:
    method: str                                      # a key of METHODS; required in version 2
    n_bands: int = 5
    encoder: str | None = None                       # one of ENCODERS; None: the method's default
    msfa_path: str | None = None                     # method "other"
    init: str | None = None
    init_std: float | None = None                    # init "normal" only
    alpha_end: float | None = None                   # inverse-temperature schedule end (softmax family)
    coef: float | None = None                        # regularization weight; formerly just entropy
    schedule: str | None = None                      # one of SCHEDULES


@dataclass(frozen=True, kw_only=True)
class PreprocessConfig:
    illuminant: str = "d65"                          # d65 | none
    observer: str = "tokyotech"                      # tokyotech | ideal | gaussian
    wls_out: tuple[int, ...] = (440, 500, 560, 620, 680)   # peaks for ideal/gaussian; tokyotech ignores it
    size: int = 256


@dataclass(frozen=True, kw_only=True)
class FinetuneConfig:
    epoch_freeze: int | None = None                  # None: no freeze, the whole run is joint


@dataclass(frozen=True, kw_only=True)
class SeedsConfig:
    """One seed per source of randomness; None: unseeded. Use a different value for each: the
    same value gives two components the same random stream. Old configs translate to
    `loader: 41`, the shuffle seed the old train.py fixed in code."""
    pattern: int | None = None                       # learned pattern's init, or the `rand` draw
    net: int | None = None                           # network init (the global RNG, seeded just before)
    loader: int | None = None                        # shuffle order


@dataclass(frozen=True, kw_only=True)
class Config:
    version: int = VERSION
    msfa: MSFAConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    revsci: RevSCIConfig = field(default_factory=RevSCIConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    seeds: SeedsConfig = field(default_factory=SeedsConfig)


SECTIONS: dict[str, type] = {"train": TrainConfig, "revsci": RevSCIConfig, "msfa": MSFAConfig,
                             "preprocess": PreprocessConfig, "finetune": FinetuneConfig,
                             "seeds": SeedsConfig}
