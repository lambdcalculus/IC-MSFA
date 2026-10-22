from torch import Tensor

from .model import MosaicModel

__all__ = ["upgrade_state_dict", "load_weights"]

_LEGACY_RAW = "0.parametrizations.msfa.original"

# keys a legacy checkpoint never stored; the model's own values stand. New-format
# checkpoints always store them, so for those nothing may be missing.
_MISSING_OK = frozenset({"encoder.generator.pattern", "encoder.generator.alpha"})


def _is_new_format(sd: dict[str, Tensor]) -> bool:
    return any(k.startswith(("encoder.", "decoder.")) for k in sd)


def upgrade_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map a checkpoint's "model" entry to MosaicModel's keys.

        new          encoder.generator.*, decoder.*         unchanged
        legacy joint 0.parametrizations.msfa.original, 1.*  tunable runs, joint phase
        legacy bare  conv1.*, layers.*, conv2.*             fixed-pattern runs, every finetune/

    A bare Rev3DCNN carries no pattern: build the model with the right FixedPattern first.
    """
    if "model" in sd and isinstance(sd["model"], dict):
        raise TypeError("got a whole checkpoint; pass its 'model' entry")
    new = [k.startswith(("encoder.", "decoder.")) for k in sd]
    if any(new):
        if not all(new):
            raise KeyError("checkpoint mixes new and legacy keys")
        return dict(sd)
    out: dict[str, Tensor] = {}
    for k, v in sd.items():
        if k == _LEGACY_RAW:
            out["encoder.generator.raw"] = v
        elif k.startswith("1."):
            out["decoder." + k[2:]] = v
        elif k.startswith("0."):
            raise KeyError(f"unknown legacy pattern key {k!r}")
        else:
            out["decoder." + k] = v
    return out


def load_weights(model: MosaicModel, sd: dict[str, Tensor]) -> None:
    """Load a checkpoint's model state, any format. Fails on anything unexpected, and on
    any missing key except, for legacy checkpoints, those they never had."""
    allowed = frozenset() if _is_new_format(sd) else _MISSING_OK
    missing, unexpected = model.load_state_dict(upgrade_state_dict(sd), strict=False)
    bad = sorted(set(missing) - allowed)
    if bad or unexpected:
        raise KeyError(f"checkpoint doesn't fit the model: missing {bad}, unexpected {unexpected}")
