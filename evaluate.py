import os
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np

from torchvision.transforms import Resize
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.functional.image import spectral_angle_mapper as sam

from dataset import MatDataset, NumpyDataset
from transforms import Illuminant, Observer, Select
from revsci.rev3dcnn_tunable import Rev3DCNN
from transforms.factory import make_ideal_cmf, make_gaussian_cmf
from transforms.transforms import ApplyMSFA
from msfa_layer import NormalizedMSFA

import argparse
import yaml
import torch
from msfa.tokyotech import RGBCO
from msfa.OSP import gen_osp

from tqdm import tqdm

from typing import Any

class IntToFloat(torch.nn.Module):
    def __init__(self, bit_width=8):
        super().__init__()
        self.bit_width = bit_width

    def forward(self, img: torch.Tensor):
        return img / (2**self.bit_width - 1)

    def __repr__(self):
        return f"{self.__class__.__name__}(bit_width={self.bit_width})"

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
    with open(f"{args.model}/config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # train_config = config.get("train", {})
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
            # get MSFA when getting model
        case "rand":
            msfa = np.load(f"{args.model}/msfa.npy")
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

    model = Rev3DCNN(n_blocks=revsci_config.get("n_blocks", 24),
                     n_split=revsci_config.get("n_split", 2)).to(device)

    if not tunable_msfa:
        data = torch.load(f"{args.model}/model_best.tar", weights_only=True)
        model.load_state_dict(data["model"])
    else:
        msfa_layer = NormalizedMSFA(n_bands,
                                    init=msfa_config.get("init", "uniform"),
                                    norm=msfa_config.get("norm", "l1"))
        model_with_mosaic = nn.Sequential(msfa_layer, model)
        epoch_freeze = config.get("finetune", {}).get("epoch_freeze", 200)
        data_pre_finetune = torch.load(f"{args.model}/model_epoch{epoch_freeze}.tar", weights_only=True)
        model_with_mosaic.load_state_dict(data_pre_finetune["model"])

        msfa_float: torch.Tensor = model_with_mosaic[0].msfa
        msfa = (msfa_float >= msfa_float.max(dim=0)[0]).float().detach().cpu()
        mosaic = ApplyMSFA(msfa)

        data_finetune = torch.load(f"{args.model}/finetune/model_best.tar", weights_only=True)
        model.load_state_dict(data_finetune["model"])

    _, test_data = create_datasets(illum, obs, resize, mosaic)

    test_loader = DataLoader(test_data,
                             batch_size=1,
                             num_workers=2,
                        )

    psnr_list = []
    sam_list = []
    ssim_list = []
    n_samples = 0
    print(f"Evaluating model: {os.path.split(args.model)[1]}")
    for inp, gt in tqdm(test_loader, desc="Evaluating..."):
        n_samples += 1
        inp: torch.Tensor = inp.to(device, non_blocking=True)
        gt: torch.Tensor = gt.to(device, non_blocking=True)

        pred = model(inp)

        psnr_list.append(psnr(pred, gt, data_range=(0.0, 1.0)).item() * inp.size(0))
        sam_list.append(sam(pred + 1e-12, gt + 1e-12, reduction="none").mean(dim=(1,2)).sum().item())
        ssim_list.append(ssim(pred, gt, data_range=(0.0, 1.0), reduction="sum").item()) # pyright: ignore[reportOperatorIssue, reportAttributeAccessIssue]

    psnr_mean, psnr_std = np.array(psnr_list).mean(), np.array(psnr_list).std()
    sam_mean, sam_std = np.array(sam_list).mean(), np.array(sam_list).std()
    ssim_mean, ssim_std = np.array(ssim_list).mean(), np.array(ssim_list).std()

    print("== EVALUATION RESULTS ==")
    print(f"  PSNR: {psnr_mean:2.3f} +- {psnr_std:2.3f}")
    print(f"  SAM:  {sam_mean:2.5f} +- {sam_std:2.5f}" )
    print(f"  SSIM: {ssim_mean:1.7f} +- {ssim_std:1.7f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, required=True, help="model folder to evaluate")

    args = parser.parse_args()
    main(args)

