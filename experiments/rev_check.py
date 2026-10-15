"""Acceptance tests for reversible training of RevSCI (branch rev-autocast).

  grad   Gradient agreement between the three backward modes of Rev3DCNN (autograd,
         checkpoint, reversible), per parameter group, on real images, under three precision
         settings: fp32; AMP with the default fp16 residual stream; AMP with `fp32_stream`.
         Covers the fixed-pattern model and the tunable joint model (MSFA logits, entropy term).
         Also reports how far the reconstructed block inputs drift from the true ones, and
         checks that an fp16 overflow under GradScaler still makes `scaler.step` skip.
  mem    Peak GPU memory and time of one training step (forward, backward, Adam step) per
         mode, on synthetic inputs; neither depends on the pixel values.
  cpu    fp64 check of the algebra on the CPU (random weights and inputs); runs anywhere.

    python experiments/rev_check.py grad --fixed-run runs/<osp5 run> --tunable-run runs/<learned5 run>
    python experiments/rev_check.py mem --bands 5 12 --sizes 256 512 1024
    python experiments/rev_check.py cpu

Without a model_best.tar in the run dir (or with --random-init) the weights are a seeded
random init. Trained weights are the realistic case for the reconstruction drift.

Peak memory is per process (torch.cuda.max_memory_allocated), so other jobs on the GPU do not
affect it; step times are affected, so run `mem` on an idle GPU when the times matter.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("Tokyotech-31B", "Manchester-33B", "CAVE-31B")   # train.py:create_datasets

SETTINGS = {                 # name: (autocast, fp32_stream)
    "fp32": (False, False),
    "amp": (True, False),
    "amp+fp32stream": (True, True),
}
MODES = ("autograd", "checkpoint", "reversible")


# ----------------------------------------------------------------------------- setup
def enter_repo() -> None:
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def check_data(cfg: dict) -> None:
    missing = [str(REPO / "data" / "datasets" / d) for d in DATASETS
               if not (REPO / "data" / "datasets" / d).is_dir()]
    if cfg.get("preprocess", {}).get("illuminant", "d65") == "d65":
        f = REPO / "data/illuminants/CIE_D65_420_720_10.mat"
        if not f.is_file():
            missing.append(str(f))
    if missing:
        sys.exit("Data not found; `grad` needs the lab machine.\n  " + "\n  ".join(missing))


def load_cfg(run_dir: Path) -> dict:
    with open(run_dir / "config.yaml") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_preprocess(cfg: dict):
    """(illum, obs, resize) exactly as train.py:main builds them."""
    import torch.nn as nn
    from torchvision.transforms import Resize
    from transforms import Illuminant, Observer
    from transforms.factory import make_ideal_cmf, make_gaussian_cmf

    pre = cfg.get("preprocess", {})
    match pre.get("illuminant", "d65"):
        case "d65":
            illum = Illuminant("data/illuminants/CIE_D65_420_720_10.mat")
        case "none":
            illum = nn.Identity()
        case other:
            raise NotImplementedError(f"illuminant {other!r}")
    wls_in = [420 + 10 * i for i in range(31)]
    wls_out = pre.get("wls_out", [440, 500, 560, 620, 680])
    match pre.get("observer", "tokyotech"):
        case "tokyotech":
            obs = Observer("data/sensitivities/C_emuda5_420_720_10.mat", transpose=True)
        case "ideal":
            obs = Observer(make_ideal_cmf(wls_in, wls_out))
        case "gaussian":
            obs = Observer(make_gaussian_cmf(wls_in, wls_out))
        case other:
            raise NotImplementedError(f"observer {other!r}")
    size = pre.get("size", 256)
    return illum, obs, Resize([size, size])


def fixed_pattern(cfg: dict, run_dir: Path) -> np.ndarray:
    m = cfg.get("msfa", {})
    match m.get("method", "tokyotech_rgbco"):
        case "tokyotech_rgbco":
            from msfa.tokyotech import RGBCO
            return np.asarray(RGBCO.msfa, dtype=float)
        case "osp":
            from msfa.OSP import gen_osp
            return gen_osp(m.get("n_bands", 5))
        case "rand" | "other" if (run_dir / "msfa.npy").is_file():
            return np.load(run_dir / "msfa.npy")
        case "other":
            return np.load(m["msfa_path"])
        case other:
            raise ValueError(f"method {other!r} is not a fixed pattern (or its msfa.npy is missing)")


def build_model(kind: str, run_dir: Path, cfg: dict, random_init: bool, seed: int):
    """nn.Sequential(front, Rev3DCNN): front is ApplyMSFA (fixed) or NormalizedMSFA (tunable)."""
    import torch
    import torch.nn as nn
    from msfa_layer import NormalizedMSFA
    from revsci.rev3dcnn_tunable import Rev3DCNN
    from transforms.transforms import ApplyMSFA

    torch.manual_seed(seed)
    r, m = cfg.get("revsci", {}), cfg.get("msfa", {})
    net = Rev3DCNN(n_blocks=r.get("n_blocks", 24), n_split=r.get("n_split", 2))
    if kind == "fixed":
        front = ApplyMSFA(torch.from_numpy(fixed_pattern(cfg, run_dir)).float())
    else:
        front = NormalizedMSFA(m.get("n_bands", 5), norm=m.get("norm", "l1"), init=m.get("init", "uniform"))
    model = nn.Sequential(front, net)

    ckpt = run_dir / "model_best.tar"
    source = f"random init (seed {seed})"
    if not random_init and ckpt.is_file():
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
        (net if kind == "fixed" else model).load_state_dict(sd)
        source = str(ckpt)
    return model, source


def load_images(cfg: dict, n: int):
    """First n training images as clean cubes; the mask is applied inside the model."""
    from train import create_datasets
    illum, obs, resize = build_preprocess(cfg)
    train_data, _ = create_datasets(illum, obs, resize)
    return [train_data[i][1] for i in range(n)]


# ----------------------------------------------------------------------------- grad
def group_of(name: str, n_blocks: int) -> list[str]:
    if name.startswith("0."):
        return ["msfa"]
    if ".conv1." in name:
        return ["conv1"]
    if ".conv2." in name:
        return ["conv2"]
    b = int(re.match(r"1\.layers\.(\d+)\.", name).group(1))   # pyright: ignore[reportOptionalMemberAccess]
    out = ["rev(all)"]
    if b == 0:
        out.append("rev[0]")
    if b == n_blocks - 1:
        out.append("rev[last]")
    return out


def compute_grads(model, kind, x, gt, amp, coef, init_scale=2.0 ** 16):
    """One backward pass as train.py does it; returns unscaled grads and the loss."""
    import torch
    from train import CustomLoss

    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", init_scale=init_scale, enabled=amp)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        pred = model(x)
        if kind == "fixed":
            loss = torch.nn.functional.l1_loss(pred, gt)
        else:
            l1, ent = CustomLoss(model[1], model[0])(pred, gt)
            loss = l1 + coef * ent
    scaler.scale(loss).backward()
    if amp:
        scaler.unscale_(torch.optim.SGD(params, lr=0.0))
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return grads, float(loss)


def rel_errors(grads, ref, n_blocks) -> dict[str, float]:
    import torch
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    worst = 0.0
    for name, r in ref.items():
        d = (grads[name].double() - r.double()).pow(2).sum().item()
        s = r.double().pow(2).sum().item()
        for g in group_of(name, n_blocks):
            num[g] = num.get(g, 0.0) + d
            den[g] = den.get(g, 0.0) + s
        if s > 0:
            worst = max(worst, (d / s) ** 0.5)
    out = {g: (num[g] / den[g]) ** 0.5 if den[g] > 0 else float("nan") for g in num}
    out["max tensor"] = worst
    out["finite"] = float(all(torch.isfinite(g).all().item() for g in grads.values()))
    return out


def reconstruction_drift(model, x, amp, fp32_stream) -> list[float]:
    """Relative error of each block's input as the reversible backward reconstructs it."""
    import torch
    net = model[1]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        h = net.conv1(model[0](x).unsqueeze(1))
        if fp32_stream:
            h = h.float()
        inputs = []
        for block in net.layers:
            inputs.append(h)
            h = block(h)
        errs = []
        for block, true in zip(reversed(net.layers), reversed(inputs)):
            h = block.reverse(h)
            errs.append(((h.float() - true.float()).norm() / true.float().norm()).item())
    return errs[::-1]


def overflow_skips(model, kind, x, gt, coef) -> tuple[bool, float]:
    """With a huge loss scale the fp16 grads overflow: scaler.step must skip and lower the scale."""
    import torch
    from train import CustomLoss

    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 40)
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        pred = model(x)
        if kind == "fixed":
            loss = torch.nn.functional.l1_loss(pred, gt)
        else:
            l1, ent = CustomLoss(model[1], model[0])(pred, gt)
            loss = l1 + coef * ent
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    skipped = all(torch.equal(b, p) for b, p in zip(before, params))
    with torch.no_grad():
        for b, p in zip(before, params):
            p.copy_(b)
    model.zero_grad(set_to_none=True)
    return skipped, scaler.get_scale()


def print_table(title, rows, cols):
    print(f"\n  {title}")
    print("  " + f"{'':<28}" + "".join(f"{c:>11}" for c in cols))
    for label, vals in rows:
        print("  " + f"{label:<28}" + "".join(
            f"{vals.get(c, float('nan')):>11.2e}" if c != "finite" else f"{('yes' if vals.get(c) else 'NO'):>11}"
            for c in cols))


def cmd_grad(a):
    import torch

    runs = [("fixed", a.fixed_run), ("tunable", a.tunable_run)]
    runs = [(k, Path(r).resolve()) for k, r in runs if r is not None]
    if not runs:
        sys.exit("pass --fixed-run and/or --tunable-run")
    enter_repo()

    for kind, run_dir in runs:
        cfg = load_cfg(run_dir)
        check_data(cfg)
        model, source = build_model(kind, run_dir, cfg, a.random_init, a.seed)
        model = model.cuda()
        net = model[1]
        n_blocks = len(net.layers)
        imgs = load_images(cfg, a.images)
        print(f"\n=== {kind}: {run_dir.name}  |  L={cfg['msfa']['n_bands']}, {n_blocks} blocks, "
              f"n_split={net.layers[0].n}, weights: {source}"
              + (f", entropy coef {a.coef}" if kind == "tunable" else ""))

        cols = (["msfa"] if kind == "tunable" else []) + ["conv1", "rev[0]", "rev(all)", "rev[last]",
                                                          "conv2", "max tensor", "finite"]
        vs_true: dict[str, list[dict]] = {}
        vs_same: dict[str, list[dict]] = {}
        drift: dict[str, list[list[float]]] = {}
        for img in imgs:
            x = img.unsqueeze(0).cuda()
            gt = x.clone()
            net.mode, net.fp32_stream = "autograd", False
            true, _ = compute_grads(model, kind, x, gt, amp=False, coef=a.coef)
            for setting, (amp, stream) in SETTINGS.items():
                net.fp32_stream = stream
                same = None
                for mode in MODES:
                    net.mode = mode
                    g, _ = compute_grads(model, kind, x, gt, amp=amp, coef=a.coef)
                    vs_true.setdefault(f"{setting} / {mode}", []).append(rel_errors(g, true, n_blocks))
                    if mode == "autograd":
                        same = g
                    else:
                        vs_same.setdefault(f"{setting} / {mode}", []).append(rel_errors(g, same, n_blocks))
                drift.setdefault(setting, []).append(reconstruction_drift(model, x, amp, stream))
            net.mode, net.fp32_stream = "autograd", False

        def summarise(d):
            # worst image per group (max), so a single bad image is not averaged away
            return [(k, {c: max(r[c] for r in v) if c != "finite" else min(r[c] for r in v) for c in v[0]})
                    for k, v in d.items()]

        print(f"\n  relative gradient error ‖g − g_ref‖/‖g_ref‖ per parameter group, worst of {len(imgs)} images")
        print_table("ref = fp32 autograd (the exact gradient, up to fp32 rounding)", summarise(vs_true), cols)
        print_table("ref = autograd under the same precision setting (what the backward mode adds)",
                    summarise(vs_same), cols)

        depths = sorted({n_blocks - 1, (3 * n_blocks) // 4, n_blocks // 2, n_blocks // 4, 0}, reverse=True)
        print(f"\n  reconstructed block input, relative error (worst image); block {n_blocks - 1} is "
              "reconstructed first, block 0 last")
        print("  " + f"{'':<28}" + "".join(f"{'block ' + str(b):>11}" for b in depths))
        for setting, per_img in drift.items():
            worst = np.max(np.array(per_img), axis=0)
            print("  " + f"{setting:<28}" + "".join(f"{worst[b]:>11.2e}" for b in depths))

        x = imgs[0].unsqueeze(0).cuda()
        print("\n  fp16 overflow under GradScaler (init_scale 2^40): step skipped? / new scale")
        for setting in ("amp", "amp+fp32stream"):
            net.fp32_stream = SETTINGS[setting][1]
            for mode in MODES:
                net.mode = mode
                skipped, scale = overflow_skips(model, kind, x, x.clone(), a.coef)
                print(f"    {setting + ' / ' + mode:<28} {'skipped' if skipped else 'STEPPED':>8}   {scale:.3g}")
        net.mode, net.fp32_stream = "autograd", False
        del model
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- cpu
def cmd_cpu(a):
    """fp64 on the CPU, random weights and inputs: checks the algebra of the reversible backward."""
    import copy
    import torch
    enter_repo()
    from revsci.rev3dcnn_tunable import Rev3DCNN
    torch.manual_seed(a.seed)
    worst = 0.0
    for n_split in (2, 4):
        for B in (1, 2):
            base = Rev3DCNN(n_blocks=4, n_split=n_split).double()
            x = torch.rand(B, 5, 16, 16, dtype=torch.float64)
            gt = torch.rand(B, 5, 16, 16, dtype=torch.float64)
            grads = {}
            for mode in MODES:
                m = copy.deepcopy(base)
                m.mode = mode
                xi = x.clone().requires_grad_()
                torch.nn.functional.l1_loss(m(xi), gt).backward()
                grads[mode] = [xi.grad] + [p.grad for p in m.parameters()]
            for mode in ("checkpoint", "reversible"):
                err = max(((g - r).norm() / r.norm()).item() for g, r in zip(grads[mode], grads["autograd"]))
                worst = max(worst, err)
                print(f"n_split={n_split} B={B} {mode:<10} max relative error over input + parameters: {err:.1e}")
    print("PASS" if worst < 1e-12 else "FAIL", f"(worst {worst:.1e}, fp64 should give < 1e-12)")


# ----------------------------------------------------------------------------- mem
def bench(n_bands, size, batch, mode, amp, stream, steps, n_blocks, n_split):
    import torch
    import torch.nn as nn
    from msfa_layer import NormalizedMSFA
    from revsci.rev3dcnn_tunable import Rev3DCNN

    torch.cuda.empty_cache()
    model = nn.Sequential(NormalizedMSFA(n_bands, norm="softmax"),
                          Rev3DCNN(n_blocks, n_split, mode=mode, fp32_stream=stream)).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    x = torch.rand(batch, n_bands, size, size, device="cuda")
    gt = torch.rand_like(x)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = torch.nn.functional.l1_loss(model(x), gt)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    try:
        step()                                   # warm-up; allocates the Adam state
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = perf_counter()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        dt = (perf_counter() - t0) / steps
        peak = torch.cuda.max_memory_allocated()
        return peak, peak - resident, dt
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        del model, opt, x, gt
        torch.cuda.empty_cache()


def cmd_mem(a):
    import torch
    enter_repo()
    torch.backends.cudnn.benchmark = False      # as in train.py (default)
    if a.mem_cap_gb is not None:
        # keeps an OOM probe here from starving another job on the same GPU
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_gb * 2**30 / total))
        print(f"memory capped at {a.mem_cap_gb} GB for this process; OOM means 'over the cap'")
    settings = ["amp", "amp+fp32stream"] + (["fp32"] if a.fp32 else [])
    # this process already holds a CUDA context (get_device_properties above), so leave it out
    others = [p for p in torch.cuda.list_gpu_processes().splitlines()[1:]
              if "process" in p and p.split()[1] != str(os.getpid())]
    if others:
        print("note: other processes are on the GPU, so step times are not reliable:\n  "
              + "\n  ".join(others))
    print(f"\n{a.blocks} blocks, n_split {a.split}, {a.steps} timed steps; "
          "peak = max_memory_allocated during a step; act = peak − resident (params, grads, Adam state)")
    print(f"{'L':>3} {'size':>5} {'B':>2} {'setting':<15} {'mode':<11} {'peak MB':>9} {'act MB':>9} {'s/step':>8}")
    for L in a.bands:
        for size in a.sizes:
            for B in a.batch:
                for setting in settings:
                    amp, stream = SETTINGS[setting]
                    for mode in MODES:
                        r = bench(L, size, B, mode, amp, stream, a.steps, a.blocks, a.split)
                        if r is None:
                            print(f"{L:>3} {size:>5} {B:>2} {setting:<15} {mode:<11} {'OOM':>9}")
                        else:
                            peak, act, dt = r
                            print(f"{L:>3} {size:>5} {B:>2} {setting:<15} {mode:<11} "
                                  f"{peak / 2**20:>9.0f} {act / 2**20:>9.0f} {dt:>8.3f}", flush=True)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("grad", help="gradient agreement on real images")
    s.add_argument("--fixed-run", help="run dir of a fixed-pattern run (config + model_best.tar)")
    s.add_argument("--tunable-run", help="run dir of a tunable run (joint-phase model_best.tar)")
    s.add_argument("--images", type=int, default=3, help="number of training images (default 3)")
    s.add_argument("--coef", type=float, default=1.0, help="entropy coefficient for the tunable model")
    s.add_argument("--random-init", action="store_true", help="ignore model_best.tar")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_grad)

    s = sub.add_parser("cpu", help="fp64 algebra check on the CPU; needs no data or GPU")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_cpu)

    s = sub.add_parser("mem", help="peak memory and step time per mode")
    s.add_argument("--bands", type=int, nargs="+", default=[5])
    s.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    s.add_argument("--batch", type=int, nargs="+", default=[1])
    s.add_argument("--blocks", type=int, default=24)
    s.add_argument("--split", type=int, default=2)
    s.add_argument("--steps", type=int, default=5)
    s.add_argument("--fp32", action="store_true", help="also measure without autocast")
    s.add_argument("--mem-cap-gb", type=float, default=None,
                   help="cap this process's GPU memory (use when another job shares the GPU)")
    s.set_defaults(fn=cmd_mem)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
