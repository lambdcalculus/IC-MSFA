"""Loading run directories written by the old train.py (before the MosaicModel restructure).
Frozen: its only job is to reproduce how old runs were evaluated (evaluate.py:124-168)."""
import csv
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from msfa_layer import FixedPattern, SoftmaxGenerator
from msfa_layer.legacy import normalized_generator
from revsci.rev3dcnn_tunable import Rev3DCNN
from .checkpoint import load_weights
from .encoders import BandSum, Encoder, MaskedCube, UnsoundEncoding
from .model import MosaicModel

__all__ = ["load_legacy_run"]

_FIXED = ("tokyotech_rgbco", "osp", "rand", "other")


def load_legacy_run(run_dir: str | Path, which: str | int = "best",
                    device: str | torch.device = "cpu") -> tuple[MosaicModel, dict]:
    """Rebuild an old run as a MosaicModel with a hard pattern, in eval mode.

    Args:
        run_dir: the run's directory (config.yaml, metrics.csv, checkpoints).
        which: "best" (model_best.tar of the last phase), "final" (the last epoch's
            checkpoint), or an overall epoch number (for tunable runs it must lie in the
            fine-tune phase; the files there restart at 1, the number here doesn't).
        device: where to put the model.

    Returns:
        (model, info), info with "checkpoint", "pattern", "epoch_freeze" (None for fixed runs).
    """
    run_dir = Path(run_dir)
    cfg = _load_config(run_dir)
    msfa_cfg = cfg.get("msfa") or {}
    method = msfa_cfg.get("method", "tokyotech_rgbco")
    encoder_cls = _encoder_class(msfa_cfg)
    epochs = (cfg.get("train") or {}).get("epochs", 300)        # train.py's default

    # Building the modules draws their (overwritten) init from the global RNG; fork it so
    # loading a run leaves the caller's random state untouched. Module init runs on the CPU.
    if method in _FIXED:
        pattern = _stored_pattern(run_dir, method, msfa_cfg)
        with torch.random.fork_rng(devices=[]):
            model = MosaicModel(encoder_cls(FixedPattern(pattern)), _decoder(cfg))
        ckpt = _checkpoint(run_dir, which, last=epochs)
        epoch_freeze = None
    elif method == "tunable":
        epoch_freeze = _freeze_epoch(run_dir, cfg)
        with torch.random.fork_rng(devices=[]), warnings.catch_warnings():
            warnings.simplefilter("ignore", UnsoundEncoding)     # soft x MaskedCube is the legacy setup
            model = MosaicModel(encoder_cls(_legacy_generator(msfa_cfg)), _decoder(cfg))
        load_weights(model, _load(run_dir / f"model_epoch{epoch_freeze}.tar"))
        _check_binarisation(model)
        model.encoder.freeze()
        ckpt = _checkpoint(run_dir / "finetune", which, last=epochs - epoch_freeze,
                           offset=epoch_freeze)
    else:
        raise ValueError(f"{run_dir}: unknown msfa.method {method!r}")

    load_weights(model, _load(ckpt))
    model.to(device).eval()
    return model, {"checkpoint": ckpt, "pattern": model.generator.hard().cpu(),
                   "epoch_freeze": epoch_freeze}


# ---------------------------------------------------------------- pieces

def _load_config(run_dir: Path) -> dict:
    path = run_dir / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; is {run_dir} a run directory?")
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def _encoder_class(msfa_cfg: dict) -> type[Encoder]:
    # the band-sum branch's key (6d616d4); absent meant the masked cube
    match msfa_cfg.get("measurement", "masked"):
        case "masked":
            return MaskedCube
        case "sum":
            return BandSum
        case other:
            raise ValueError(f"unknown msfa.measurement {other!r}")


def _decoder(cfg: dict) -> Rev3DCNN:
    r = cfg.get("revsci") or {}
    return Rev3DCNN(n_blocks=r.get("n_blocks", 24), n_split=r.get("n_split", 2))


def _legacy_generator(msfa_cfg: dict):
    gen = normalized_generator(msfa_cfg.get("n_bands", 5),
                               norm=msfa_cfg.get("norm", "l1"),
                               init=msfa_cfg.get("init", "uniform"))
    # band-sum-temp (db28249) binarised softmax(alpha_end * logits) at the freeze; alpha
    # doesn't move the argmax, but ties can only be checked at the alpha that was used
    alpha_end = msfa_cfg.get("alpha_end")
    if alpha_end is not None:
        if not isinstance(gen, SoftmaxGenerator):
            raise ValueError("msfa.alpha_end needs norm: softmax")
        gen.alpha.fill_(float(alpha_end))
    return gen


def _stored_pattern(run_dir: Path, method: str, msfa_cfg: dict) -> np.ndarray:
    """The pattern a fixed run used, from a file. Never regenerated from code: if a pattern
    recipe (e.g. msfa/OSP.py) changes, old runs would silently be evaluated against a
    different pattern."""
    path = run_dir / "msfa.npy"                          # written by train.py for "rand", backfilled for the rest
    if not path.is_file() and method == "other" and msfa_cfg.get("msfa_path"):
        path = Path(msfa_cfg["msfa_path"])               # a stored file, but outside the run dir
        warnings.warn(f"{run_dir}: no msfa.npy, using msfa_path {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir}: no stored pattern (msfa.npy) for method {method!r}. "
                                "Old fixed-pattern runs need their pattern saved there first.")
    pattern = np.load(path)
    # the decoder's weights don't depend on L, so a wrong pattern file would load silently
    n_bands = msfa_cfg.get("n_bands", 5)
    if pattern.ndim != 3 or pattern.shape[0] != n_bands:
        raise ValueError(f"{path}: pattern shape {pattern.shape}, but the config has n_bands {n_bands}")
    return pattern


def _freeze_epoch(run_dir: Path, cfg: dict) -> int:
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
    stated = (cfg.get("finetune") or {}).get("epoch_freeze")
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
