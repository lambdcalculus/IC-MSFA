"""Old config files (no `version` key) -> version-2 dicts.

Frozen: it encodes what the old train.py did with each key, its defaults included. They are
written out explicitly, so a legacy file keeps its meaning even if a version-2 default changes.
Keys the old train.py ignored are dropped.
"""
import copy
from typing import Any

from .schema import ConfigError

__all__ = ["translate"]

# old train.py's defaults (train.py:main and TrainerConfig)
_TRAIN_DEFAULTS = {"epochs": 300, "lr": 2e-4, "decay_factor": 0.995, "amp": True, "batch_train": 1,
                   "batch_test": 1, "workers_train": 3, "workers_test": 3, "checkpoint_interval": 25,
                   "metrics_report_interval": 10}
_REVSCI_DEFAULTS = {"n_blocks": 24, "n_split": 2}
_PREPROCESS_DEFAULTS = {"illuminant": "d65", "observer": "tokyotech",
                        "wls_out": [440, 500, 560, 620, 680], "size": 256}
_FIXED_METHODS = ("tokyotech_rgbco", "osp", "rand", "other")
# msfa.measurement, from the band-sum experiments; absent meant the masked cube
_MEASUREMENT_TO_ENCODER = {None: "masked_cube", "masked": "masked_cube", "sum": "band_sum"}


def translate(raw: dict[str, Any], source: str) -> dict[str, Any]:
    d = copy.deepcopy(raw)
    out: dict[str, Any] = {"version": 2}
    for name, defaults in (("train", _TRAIN_DEFAULTS), ("revsci", _REVSCI_DEFAULTS),
                           ("preprocess", _PREPROCESS_DEFAULTS)):
        out[name] = {**defaults, **_pop_section(d, name, source)}
    # the old train.py read rev_mode and did nothing with it, but a run that set it may come from
    # a train.py that did (a reversible backward)
    if out["train"].pop("rev_mode", False):
        raise ConfigError(f"{source}: train.rev_mode: true isn't supported yet")
    out["finetune"] = {"epoch_freeze": None, **_pop_section(d, "finetune", source)}
    out["msfa"] = _msfa(_pop_section(d, "msfa", source), source)
    out["seeds"] = {"loader": 41}                    # the old train.py's shuffle seed; init was unseeded
    out.update(d)                                    # anything else is left for the loader to reject
    return out


def _pop_section(d: dict[str, Any], name: str, source: str) -> dict[str, Any]:
    values = d.pop(name, None)
    if values is None:                               # absent, or a header with nothing under it
        return {}
    if not isinstance(values, dict):
        raise ConfigError(f"{source}: {name}: expected a mapping, got {type(values).__name__}")
    return values


def _msfa(old: dict[str, Any], source: str) -> dict[str, Any]:
    method = old.pop("method", "tokyotech_rgbco")
    norm = old.pop("norm", None)
    ramp = old.pop("ramp", False)
    measurement = old.pop("measurement", None)
    path = old.pop("msfa_path", None)                # read only by method "other"
    if measurement not in _MEASUREMENT_TO_ENCODER:
        raise ConfigError(f"{source}: unknown msfa.measurement {measurement!r}")
    new: dict[str, Any] = {"n_bands": old.pop("n_bands", 5)}

    if method == "tunable":
        if norm not in (None, "l1", "softmax"):
            raise ConfigError(f"{source}: unknown msfa.norm {norm!r}")
        new["method"] = "legacy_softmax" if norm == "softmax" else "legacy_l1"
        new["encoder"] = _MEASUREMENT_TO_ENCODER[measurement]
        new["init"] = old.pop("init", "uniform")
        new["coef"] = old.pop("coef", 0.0)
        new["schedule"] = "legacy_ramp" if ramp else "constant"
        if "alpha_end" in old:                       # softmax annealing, from the band-sum experiments
            new["alpha_end"] = old.pop("alpha_end")
    elif method in _FIXED_METHODS:
        new["method"] = method
        new["encoder"] = "masked_cube"               # measurement only ever reached the tunable layer
        for key in ("init", "coef", "alpha_end"):    # the old train.py ignored these for fixed patterns
            old.pop(key, None)
        if method == "other":
            new["msfa_path"] = path
    else:
        raise ConfigError(f"{source}: unknown msfa.method {method!r} (no `version` key, so it was "
                          "read as an old config)")

    new.update(old)                                  # leftovers: the loader reports them
    return new
