import csv
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np

from torchvision.transforms import Resize
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.functional.image import spectral_angle_mapper as sam

from dataclasses import dataclass

from dataset import MatDataset, NumpyDataset
from msfa.rand import rand_msfa
from transforms import Illuminant, Observer, Select
from revsci.rev3dcnn_tunable import Rev3DCNN
from transforms.factory import make_ideal_cmf, make_gaussian_cmf
from transforms.transforms import ApplyMSFA
from msfa_layer import NormalizedMSFA

COEF = 0.01

def coef_func(epoch, _):
    return np.power(0.995, -epoch)

class CustomLoss(nn.Module):
    def __init__(self, revsci: Rev3DCNN, msfa_layer: NormalizedMSFA):
        super().__init__()
        self.revsci = revsci
        self.msfa_layer = msfa_layer

        self.l1_rev = nn.L1Loss()
        # self.l1_msfa = nn.L1Loss()

    def forward(self, pred, gt):
        rev_loss = self.l1_rev(pred, gt)
        entropy_loss = - (self.msfa_layer.msfa * torch.log(self.msfa_layer.msfa)).sum(dim=0).mean()

        return rev_loss, entropy_loss

@dataclass
class TrainerConfig:
    epochs: int
    out_dir: str

    amp: bool = True

    # optim: adam
    lr: float =  2e-4
    # scheduler: exponential
    decay_factor: float = 0.995

    # loss: allow l1 or custom loss

    checkpoint_interval: int = 25
    metrics_report_interval: int = 10
    
    rev_mode: bool = False
    tunable_msfa: bool = False

    coef: float = 0.0
    ramp: bool = False

class Trainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # wandb_run: Optional[wandb.Run] = None
    ):
        self.cfg = cfg

        self.device = device
        self.model = model.to(device)

        if self.cfg.tunable_msfa:
            # model is nn.Sequential
            self.loss_fn = CustomLoss(self.model[1], self.model[0]) # pyright: ignore[reportIndexIssue] 
        else:
            self.loss_fn = nn.L1Loss().to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.scaler = torch.cuda.amp.GradScaler() if (self.cfg.amp and self.device.type == "cuda") else None

        self.optim = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, self.cfg.decay_factor, -1)
    
        self.out_dir = Path(self.cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_out_path = self.out_dir / "model_best.tar"
        self.csv = self.out_dir / "metrics.csv"

        if not self.csv.exists():
            with open(self.csv, "w", newline="") as f:
                w = csv.writer(f)
                header = [
                    "epoch", "lr",
                    "train_loss", "train_l1_loss", "train_entropy_loss",
                    "val_loss", "val_l1_loss", "val_entropy_loss",
                    "PSNR", "SAM", "SSIM",
                    ]
                w.writerow(header)

        self.best_val = float("inf")

    def _current_lr(self) -> float:
        if self.optim.param_groups:
            return float(self.optim.param_groups[0].get("lr", 0.0))
        return 0.0

    def _log_csv(self, epoch: int, train_stats: dict[str, float], val_stats: dict[str, float]):
        row = [
            epoch,
            self._current_lr(),
            train_stats.get("train_loss", float("nan")),
            train_stats.get("l1_loss", " "),
            train_stats.get("entropy_loss", " "),
            val_stats.get("val_loss", float("nan")),
            train_stats.get("l1_loss", " "),
            train_stats.get("entropy_loss", " "),
            val_stats.get("psnr", " "),
            val_stats.get("sam", " "),
            val_stats.get("ssim", " "),
        ]
        with open(self.csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
        }
        if is_best:
            torch.save(state, self.best_out_path)
            print(f"[Saved BEST model @ epoch {epoch}] → {self.best_out_path}")
        if epoch % self.cfg.checkpoint_interval == 0:
            torch.save(state, self.out_dir / f"model_epoch{epoch}.tar")

    def _print_epoch(self, epoch: int, train_stats: dict[str, float], val_stats: dict[str, float]):
        parts = [
            f"[{epoch:03d}] epoch finished |",
            f"lr: {self._current_lr():.2e}",
            f"train: {train_stats.get('train_loss', float('nan')):7.4f}",
            f"val: {val_stats.get('val_loss', float('nan')):7.4f}"
        ]

        print(" ".join(parts))

    def train_epoch(self, epoch: int):
        self.model.train()

        running = 0.0

        # only used if tunable msfa
        running_l1 = 0.0
        running_entropy = 0.0

        n_samples = 0
        total_samples = len(self.train_loader.dataset) # pyright: ignore[reportArgumentType]

        total_grad_time = 0.0
        total_time_start = perf_counter()
        for inp, gt in self.train_loader:
            inp: torch.Tensor = inp.to(self.device, non_blocking=True)
            gt: torch.Tensor = gt.to(self.device, non_blocking=True)

            self.optim.zero_grad(set_to_none=True)
            if self.scaler is None:
                pred = self.model(inp)

                if not self.cfg.tunable_msfa:
                    loss = self.loss_fn(pred, gt)
                    loss.backward()
                else:
                    l1_loss, ent_loss = self.loss_fn(pred, gt)

                    if not self.cfg.ramp:
                        loss = l1_loss + self.cfg.coef * ent_loss
                        l1_loss.backward()
                        (self.cfg.coef * ent_loss).backward()
                    else:
                        coef = self.cfg.coef * coef_func(epoch, self.cfg.epochs)

                        loss = l1_loss + coef * ent_loss
                        l1_loss.backward()
                        (coef * ent_loss).backward()

                grad_time_start = perf_counter()
                if self.device.type == "cuda":
                    # synchronize for timing accuracy
                    torch.cuda.synchronize(self.device)
                grad_time = perf_counter() - grad_time_start
                total_grad_time += grad_time

                self.optim.step()
            else:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    pred = self.model(inp)
                    if self.cfg.tunable_msfa:
                        l1_loss, ent_loss = self.loss_fn(pred, gt)

                        if not self.cfg.ramp:
                            loss = l1_loss + self.cfg.coef * ent_loss
                        else:
                            coef = self.cfg.coef * coef_func(epoch, self.cfg.epochs)
                            loss = l1_loss + coef * ent_loss
                    else:
                        loss = self.loss_fn(pred, gt)
                self.scaler.scale(loss).backward()

                grad_time_start = perf_counter()
                # synchronize for timing accuracy
                torch.cuda.synchronize(self.device)
                grad_time = perf_counter() - grad_time_start
                total_grad_time += grad_time

                self.scaler.step(self.optim)
                self.scaler.update()

            running += float(loss.item()) * inp.size(0)
            if self.cfg.tunable_msfa:
                running_l1 += float(l1_loss.item()) * inp.size(0)       # pyright: ignore[reportPossiblyUnboundVariable]
                running_entropy += float(ent_loss.item()) * inp.size(0) # pyright: ignore[reportPossiblyUnboundVariable]

            n_samples += inp.size(0)
            if n_samples > 0:
                if not self.cfg.tunable_msfa:
                    print(f"[{epoch:03d}] (training...) running avg loss: {(running / n_samples):7.6f}   [ {n_samples:3d} / {total_samples:3d}]", end = '\r')
                else:
                    print(f"[{epoch:03d}] (training...) running avg loss: {(running / n_samples):7.6f},",
                          f"l1_loss: {(running_l1 / n_samples):7.6f}",
                          f"entropy_loss: {(running_entropy / n_samples):7.6f}",
                          f"[ {n_samples:3d} / {total_samples:3d}]", end = '\r')


        # if self.wandb_run is not None:
        #     self.wandb_run.log({"epoch": epoch, "lr": self._current_lr(), "train loss": avg})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        total_time = perf_counter() - total_time_start
        print(f"\n[{epoch:03d}] training finished.",
              f"Total time: {total_time:.3f}s",
              f"/ Train time: {total_grad_time:.3f}s",
              f"/ I/O time: {(total_time-total_grad_time):.3f}s")

        out = {"train_loss": running / max(n_samples, 1)}
        if self.cfg.tunable_msfa:
            out["l1_loss"] = running_l1 / max(n_samples, 1)
            out["entropy_loss"] = running_entropy / max(n_samples, 1)

        return out

    @torch.no_grad()
    def validate(self, epoch: int, validate_metrics: bool):
        self.model.eval()

        loss_sum = 0.0

        # only used when tunable msfa
        l1_sum = 0.0
        entropy_sum = 0.0

        n_samples = 0
        total_samples = len(self.val_loader.dataset) # pyright: ignore[reportArgumentType]

        total_val_time = 0.0
        total_time_start = perf_counter()

        val_psnr, val_sam, val_ssim = 0.0, 0.0, 0.0

        for inp, gt in self.val_loader:
            inp: torch.Tensor = inp.to(self.device, non_blocking=True)
            gt:  torch.Tensor = gt.to(self.device, non_blocking=True)

            # forward (no grad)
            with torch.no_grad():
                pred = self.model(inp).clamp(0, 1)
                if not self.cfg.tunable_msfa:
                    loss = self.loss_fn(pred, gt)
                else:
                    l1_loss, entropy_loss = self.loss_fn(pred, gt)

                    if not self.cfg.ramp:
                        loss = l1_loss + self.cfg.coef * entropy_loss
                    else:
                        coef = self.cfg.coef * coef_func(epoch, self.cfg.epochs)
                        loss = l1_loss + coef * entropy_loss

            val_time_start = perf_counter()
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            loss_sum += float(loss.item()) * inp.size(0)
            if self.cfg.tunable_msfa:
                l1_sum += float(l1_loss.item()) * inp.size(0)           # pyright: ignore[reportPossiblyUnboundVariable]
                entropy_sum += float(entropy_loss.item()) * inp.size(0) # pyright: ignore[reportPossiblyUnboundVariable]

            n_samples += inp.size(0)

            if n_samples > 0:
                print(f"[{epoch:03d}] (validating...) running avg loss: {(loss_sum / n_samples):7.6f}   [ {n_samples:3d} / {total_samples:3d}]", end = '\r')

            if validate_metrics:
                val_psnr += psnr(pred, gt, data_range=(0.0, 1.0)).item() * inp.size(0)
                val_sam  += sam(pred + 1e-12, gt + 1e-12, reduction="none").mean(dim=(1,2)).sum().item()
                val_ssim += ssim(pred, gt, data_range=(0.0, 1.0), reduction="sum").item() # pyright: ignore[reportOperatorIssue, reportAttributeAccessIssue]

            val_time = perf_counter() - val_time_start
            total_val_time += val_time

        if validate_metrics:
            val_psnr /= n_samples
            val_sam /= n_samples
            val_ssim /= n_samples
        total_time = perf_counter() - total_time_start

        print()
        if validate_metrics:
            print(f"[{epoch:03d}] Test PSNR: {val_psnr:2.3f}")
            print(f"[{epoch:03d}] Test SAM:  {val_sam:2.5f}")
            print(f"[{epoch:03d}] Test SSIM: {val_ssim:1.7f}")
        print(f"[{epoch:03d}] validation finished.",
              f"Time: {total_time:.3f}s",
              f"/ Val time: {total_val_time:.3f}s",
              f"/ I/O time: {(total_time-total_val_time):.3f}s")

        out = {"val_loss": (loss_sum / n_samples)}
        if validate_metrics:
            out["psnr"] = val_psnr
            out["sam"] = val_sam
            out["ssim"] = val_ssim
        if self.cfg.tunable_msfa:
            out["l1_loss"] = l1_sum / n_samples
            out["entropy_loss"] = entropy_sum / n_samples

        return out

    def fit(self):
        print(f"Start training for {self.cfg.epochs} epochs. Logs → {self.csv}")
        for ep in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch(ep)

            validate_metrics: bool = (ep % self.cfg.metrics_report_interval) == 0 if \
                                     self.cfg.metrics_report_interval > 0 else False
            val_stats = self.validate(ep, validate_metrics=validate_metrics)

            self._print_epoch(ep, train_loss, val_stats)
            self._log_csv(ep, train_loss, val_stats)

            is_best = (val_stats["val_loss"] < self.best_val) if not self.cfg.tunable_msfa \
                      else (val_stats["l1_loss"] < self.best_val)
            if is_best:
                self.best_val = val_stats["val_loss"] if not self.cfg.tunable_msfa else val_stats["l1_loss"]
            self._save_checkpoint(ep, is_best)

# TODO: add this to transforms
class IntToFloat(torch.nn.Module):
    def __init__(self, bit_width=8):
        super().__init__()
        self.bit_width = bit_width

    def forward(self, img):
        assert not img.is_floating_point(), "Trying to convert already normalized tensor."
        return img / (2**self.bit_width - 1)

    def __repr__(self):
        return f"{self.__class__.__name__}(bit_width={self.bit_width})"

from msfa.tokyotech import RGBCO
from msfa.OSP import gen_osp
import os
import argparse
import yaml
from datetime import datetime
from shutil import copy2
from typing import Any

def create_datasets(illum, obs, resize, mosaic=None):
    if mosaic is not None:
        transform_x = nn.Sequential(illum, obs, resize, mosaic)
        transform_y = nn.Sequential(illum, obs, resize)
    else:
        transform_x = nn.Sequential(illum, obs, resize)
        transform_y = nn.Sequential(illum, obs, resize)

    # pre_transform_ttech = not required
    pre_transform_cave = IntToFloat(16)
    pre_transform_man = Select([i for i in range(31)])

    ttech_transform_x, ttech_transform_y = transform_x, transform_y
    cave_transform_x, cave_transform_y = (
            nn.Sequential(pre_transform_cave, transform_x),
            nn.Sequential(pre_transform_cave, transform_y)
    )
    man_transform_x, man_transform_y = (
            nn.Sequential(pre_transform_man, transform_x),
            nn.Sequential(pre_transform_man, transform_y)
    )

    tokyotech = MatDataset(
            "./data/datasets/Tokyotech-31B",
            transform_x=ttech_transform_x,
            transform_y=ttech_transform_y
    )
    manchester = MatDataset(
            "./data/datasets/Manchester-33B",
            transform_x=man_transform_x,
            transform_y=man_transform_y
    )
    cave = NumpyDataset(
            "./data/datasets/CAVE-31B",
            transform_x=cave_transform_x,
            transform_y=cave_transform_y
    )

    # TODO: add seed to CLI
    gen = torch.Generator().manual_seed(41)
    all_data = ConcatDataset([tokyotech, manchester, cave])
    train_data, test_data = torch.utils.data.random_split(all_data, [0.8, 0.2], generator=gen)
    return train_data, test_data


def main(args):
    config: dict[str, Any]
    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config_name = os.path.splitext(os.path.split(args.config)[1])[0]
    run_name = f"{config_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir = f"runs/{run_name}"
    os.makedirs(out_dir)

    copy2(args.config, f"{out_dir}/config.yaml")

    train_config = config.get("train", {})
    revsci_config = config.get("revsci", {})
    msfa_config = config.get("msfa", {})
    preprocess_config = config.get("preprocess", {})

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU: {torch.cuda.get_device_name(device)}")

    match preprocess_config.get("illuminant", "d65"):
        case "d65":
            illum = Illuminant('data/illuminants/CIE_D65_420_720_10.mat')
        case "none":
            illum = nn.Identity()
        case _:
            raise NotImplementedError()

    WLS_IN  = [420 + 10*i for i in range(31)]
    wls_out: list[int] = preprocess_config.get("wls_out", [440, 500, 560, 620, 680])
    match preprocess_config.get("observer", "tokyotech"):
        case "tokyotech":
            obs = Observer("data/sensitivities/C_emuda5_420_720_10.mat", transpose=True)
        case "ideal":
            obs = Observer(make_ideal_cmf(WLS_IN, wls_out))
        case "gaussian":
            obs = Observer(make_gaussian_cmf(WLS_IN, wls_out))
        case _:
            raise NotImplementedError()

    n_bands = msfa_config.get("n_bands", 5)
    if obs.cmf.shape[0] != n_bands:
        raise ValueError(f"n_bands is {n_bands} but observer CMF outputs {obs.cmf.shape[0]}, they should be identical.")

    tunable_msfa = False
    match msfa_config.get("method", "tokyotech_rgbco"):
        case "tokyotech_rgbco":
            mosaic = ApplyMSFA(torch.from_numpy(RGBCO.msfa).float())
        case "osp":
            msfa = gen_osp(n_bands)
            mosaic = ApplyMSFA(torch.from_numpy(msfa).float())
        case "tunable":
            tunable_msfa = True
            mosaic = NormalizedMSFA(
                    n_bands,
                    norm=msfa_config.get("norm", "l1"),
                    init=msfa_config.get("init", "uniform"),
                    # "sum": proposal eq. 2; absent: the masked cube, as in the FAPESP runs
                    measurement=msfa_config.get("measurement", "masked")
                    )
        case "rand":
            msfa = rand_msfa(n_bands)
            np.save(f"{out_dir}/msfa.npy", msfa)
            mosaic = ApplyMSFA(torch.from_numpy(msfa).float())
        case "other":
            path = msfa_config.get("msfa_path", None)
            if path is None:
                raise ValueError(f"method is 'other' but 'msfa_path' wasn't specified")
            msfa = np.load(msfa_config.get("msfa_path", None))
            mosaic = ApplyMSFA(torch.from_numpy(msfa).float())
        case _:
            raise NotImplementedError()


    size = preprocess_config.get("size", 256)
    resize = Resize([size, size])

    if not tunable_msfa:
        train_data, test_data = create_datasets(illum, obs, resize, mosaic)
    else:
        train_data, test_data = create_datasets(illum, obs, resize)

    gen = torch.Generator().manual_seed(41)
    train_loader = DataLoader(train_data,
                              batch_size=train_config.get("batch_train", 1),
                              num_workers=train_config.get("workers_train", 3),
                              shuffle=True,
                              generator=gen,
                              pin_memory=True,
                              persistent_workers=True
                        )
    test_loader = DataLoader(test_data,
                             batch_size=train_config.get("batch_test", 1),
                             num_workers=train_config.get("workers_test", 3),
                        )

    revsci = Rev3DCNN(n_blocks=revsci_config.get("n_blocks", 24),
                     n_split=revsci_config.get("n_split", 2),
                     mode="reversible" if train_config.get("rev_mode", False) else "autograd",
                     # absent: Rev3DCNN decides (fp32 stream only in reversible mode)
                     fp32_stream=revsci_config.get("fp32_stream")).to(device)
    if not tunable_msfa:
        model = revsci
    else:
        model = nn.Sequential(mosaic, revsci)
    
    finetune_config = config.get("finetune", {})
    epoch_freeze = finetune_config.get("epoch_freeze", None)

    # TODO: add out_dirs, logging
    if epoch_freeze is None:
        cfg = TrainerConfig(
                epochs=train_config.get("epochs", 300),
                out_dir=out_dir,
                amp=train_config.get("amp", True),
                lr=train_config.get("lr", 2e-4),
                decay_factor=train_config.get("decay_factor", 0.995),
                checkpoint_interval=train_config.get("checkpoint_interval", 25),
                metrics_report_interval=train_config.get("metrics_report_interval", 10),
                rev_mode=train_config.get("rev_mode", False),
                tunable_msfa=tunable_msfa,
                coef=msfa_config.get("coef", 0.0),
                ramp=msfa_config.get("ramp", False)
                )

        trainer = Trainer(cfg, model, train_loader, test_loader, device)

        trainer.fit()
    else:
        cfg = TrainerConfig(
                    epochs=epoch_freeze,
                    out_dir=out_dir,
                    amp=train_config.get("amp", True),
                    lr=train_config.get("lr", 2e-4),
                    decay_factor=train_config.get("decay_factor", 0.995),
                    checkpoint_interval=train_config.get("checkpoint_interval", 25),
                    metrics_report_interval=train_config.get("metrics_report_interval", 10),
                    rev_mode=train_config.get("rev_mode", False),
                    tunable_msfa=tunable_msfa,
                    coef=msfa_config.get("coef", 0.0),
                    ramp=msfa_config.get("ramp", False)
                    )

        trainer_pre = Trainer(cfg, model, train_loader, test_loader, device)

        trainer_pre.fit()

        msfa_float = mosaic.msfa
        msfa = (msfa_float >= msfa_float.max(dim=0)[0]).float().detach().cpu()

        train_new, test_new = create_datasets(illum, obs, resize, ApplyMSFA(msfa))
        train_loader = DataLoader(train_new,
                                  batch_size=train_config.get("batch_train", 1),
                                  num_workers=train_config.get("workers_train", 3),
                                  shuffle=True,
                                  generator=gen,
                                  pin_memory=True,
                                  persistent_workers=True
                            )
        test_loader = DataLoader(test_new,
                                 batch_size=train_config.get("batch_test", 1),
                                 num_workers=train_config.get("workers_test", 3),
                            )

        cfg.epochs = train_config.get("epochs", 300) - epoch_freeze
        cfg.out_dir = f"{out_dir}/finetune"
        cfg.tunable_msfa = False

        trainer_finetune = Trainer(cfg, revsci, train_loader, test_loader, device)
        trainer_finetune.optim = trainer_pre.optim
        trainer_finetune.lr_scheduler = trainer_pre.lr_scheduler
        trainer_finetune.scaler = trainer_pre.scaler

        trainer_finetune.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=True, help="config file to use")

    args = parser.parse_args()
    main(args)

