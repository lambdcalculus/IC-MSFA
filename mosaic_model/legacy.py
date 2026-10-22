"""Loading run directories written by the old train.py (before the MosaicModel restructure).
Frozen: its only job is to reproduce how old runs were evaluated (evaluate.py:124-168)."""
import csv
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from msfa_layer import FixedPattern, PatternGenerator, SoftmaxGenerator
from msfa_layer.legacy import normalized_generator
from revsci.rev3dcnn_tunable import Rev3DCNN
from runconfig import Config, MSFAConfig, load_config
from .checkpoint import load_weights
from .encoders import BandSum, Encoder, MaskedCube, UnsoundEncoding
from .model import MosaicModel

__all__ = ["load_legacy_run"]

_FIXED_METHODS = ("tokyotech_rgbco", "osp", "rand", "other")
# the old `tunable` method, split by its norm
_METHOD_TO_NORM: dict[str, Literal["l1", "softmax"]] = {"legacy_l1": "l1", "legacy_softmax": "softmax"}
_ENCODER_CLASSES: dict[str, type[Encoder]] = {"masked_cube": MaskedCube, "band_sum": BandSum}


def load_legacy_run(run_dir: str | Path, which: str | int = "best",
                    device: str | torch.device = "cpu") -> tuple[MosaicModel, dict]:
    """Rebuild an old run as a MosaicModel with a hard pattern, in eval mode.

    Args:
        run_dir: the run's directory (config.yaml, metrics.csv, checkpoints).
        which: "best" (model_best.tar of the last phase), "final" (the last epoch's
            checkpoint), or an overall epoch number (for tunable runs it must lie in the
            fine-tune phase; the files there restart at 1, the number here doesn't).
            Fixed-pattern runs only use the top-level checkpoints, as evaluate.py did.
        device: where to put the model.

    Returns:
        (model, info), info with "checkpoint", "pattern", "epoch_freeze" (None for fixed runs)
        and "config" (the run's config, resolved).
    """
    run_dir = Path(run_dir)
    cfg = _load_config(run_dir)
    m = cfg.msfa
    assert m.encoder is not None                         # loading fills it in
    encoder_cls = _ENCODER_CLASSES[m.encoder]

    # Building the modules draws their (overwritten) init from the global RNG; fork it so
    # loading a run leaves the caller's random state untouched. Module init runs on the CPU.
    if m.method in _FIXED_METHODS:
        pattern = _stored_pattern(run_dir, m)
        with torch.random.fork_rng(devices=[]):
            model = MosaicModel(encoder_cls(FixedPattern(pattern)), _decoder(cfg))
        if (run_dir / "finetune").is_dir():
            warnings.warn(f"{run_dir}: fixed-pattern run with a finetune/ phase; using the "
                          "first phase's checkpoints, as evaluate.py did")
        first_phase = cfg.finetune.epoch_freeze or cfg.train.epochs
        ckpt = _checkpoint(run_dir, which, last=first_phase)
        epoch_freeze = None
    elif m.method in _METHOD_TO_NORM:
        epoch_freeze = _freeze_epoch(run_dir, cfg)
        with torch.random.fork_rng(devices=[]), warnings.catch_warnings():
            warnings.simplefilter("ignore", UnsoundEncoding)     # soft x MaskedCube is the legacy setup
            model = MosaicModel(encoder_cls(_legacy_generator(m)), _decoder(cfg))
        load_weights(model, _load(run_dir / f"model_epoch{epoch_freeze}.tar"))
        _check_binarisation(model)
        model.encoder.freeze()
        ckpt = _checkpoint(run_dir / "finetune", which, last=cfg.train.epochs - epoch_freeze,
                           offset=epoch_freeze)
    else:                                                # a version-2 method
        raise ValueError(f"{run_dir}: method {m.method!r} isn't one the old train.py had")

    load_weights(model, _load(ckpt))
    model.to(device).eval()
    return model, {"checkpoint": ckpt, "pattern": model.generator.hard().cpu(),
                   "epoch_freeze": epoch_freeze, "config": cfg}


# ---------------------------------------------------------------- pieces

def _load_config(run_dir: Path) -> Config:
    path = run_dir / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; is {run_dir} a run directory?")
    return load_config(path)


def _decoder(cfg: Config) -> Rev3DCNN:
    return Rev3DCNN(n_blocks=cfg.revsci.n_blocks, n_split=cfg.revsci.n_split)


def _legacy_generator(m: MSFAConfig) -> PatternGenerator:
    assert m.init in ("uniform", "constant")             # loading checked it; narrows the type
    gen = normalized_generator(m.n_bands, norm=_METHOD_TO_NORM[m.method], init=m.init)
    # the band-sum experiments binarised softmax(alpha_end * logits) at the freeze; alpha
    # doesn't move the argmax, but ties can only be checked at the alpha that was used
    if m.alpha_end is not None:
        assert isinstance(gen, SoftmaxGenerator)         # loading allows it only for legacy_softmax
        gen.alpha.fill_(m.alpha_end)
    return gen


def _stored_pattern(run_dir: Path, m: MSFAConfig) -> np.ndarray:
    """The pattern a fixed run used, from a file. Never regenerated from code: if a pattern
    recipe (e.g. msfa/OSP.py) changes, old runs would silently be evaluated against a
    different pattern."""
    path = run_dir / "msfa.npy"                          # written by train.py for "rand", backfilled for the rest
    if not path.is_file() and m.msfa_path is not None:   # only method "other" keeps a path
        path = Path(m.msfa_path)                         # a stored file, but outside the run dir
        warnings.warn(f"{run_dir}: no msfa.npy, using msfa_path {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir}: no stored pattern (msfa.npy) for method {m.method!r}. "
                                "Old fixed-pattern runs need their pattern saved there first.")
    pattern = np.load(path)
    # the decoder's weights don't depend on L, so a wrong pattern file would load silently
    if pattern.ndim != 3 or pattern.shape[0] != m.n_bands:
        raise ValueError(f"{path}: pattern shape {pattern.shape}, but the config has n_bands {m.n_bands}")
    return pattern


def _freeze_epoch(run_dir: Path, cfg: Config) -> int:
    """The joint phase's length, from the run dir: the top-level metrics.csv has one row
    per joint epoch (the fine-tune writes finetune/metrics.csv)."""
    if not (run_dir / "finetune").is_dir():
        raise ValueError(f"{run_dir}: no finetune/, so this tunable run never froze")
    metrics = run_dir / "metrics.csv"
    if not metrics.is_file():
        raise FileNotFoundError(f"{metrics} not found; it gives the freeze epoch")
    with open(metrics, newline="") as f:
        logged = [int(row["epoch"]) for row in csv.DictReader(f)]
    if logged != list(range(1, len(logged) + 1)):
        raise ValueError(f"{run_dir}/metrics.csv: epochs aren't 1..N (appended rerun?)")
    on_disk = len(logged)
    stated = cfg.finetune.epoch_freeze
    if stated is None:
        warnings.warn(f"{run_dir}: config has no finetune.epoch_freeze; "
                      f"using {on_disk} from metrics.csv")
        return on_disk
    if stated != on_disk:
        raise ValueError(f"{run_dir}: epoch_freeze {stated} in config, {on_disk} in metrics.csv")
    return stated


@torch.no_grad()
def _check_binarisation(model: MosaicModel) -> None:
    """The old freeze binarised with >= max (train.py:583), the new one with argmax.
    They differ only on a tie; raise rather than pick silently."""
    p = model.generator.probs()
    legacy = (p >= p.max(dim=0)[0]).float()
    if not torch.equal(legacy, model.generator.hard()):
        raise ValueError("pattern at the freeze has a tie: old (>= max) and new (argmax) "
                         "binarisation differ")


def _checkpoint(phase_dir: Path, which: str | int, last: int, offset: int = 0) -> Path:
    if isinstance(which, bool):                          # bool is an int subclass
        raise ValueError(f"which must be 'best', 'final' or an epoch, got {which!r}")
    match which:
        case "best":
            path = phase_dir / "model_best.tar"
        case "final":
            path = phase_dir / f"model_epoch{last}.tar"
        case int() if offset < which <= offset + last:
            path = phase_dir / f"model_epoch{which - offset}.tar"
        case int():
            raise ValueError(f"epoch {which} is outside {phase_dir.name}'s epochs "
                             f"{offset + 1}..{offset + last}")
        case _:
            raise ValueError(f"which must be 'best', 'final' or an epoch, got {which!r}")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return path


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return torch.load(path, map_location="cpu", weights_only=True)["model"]
