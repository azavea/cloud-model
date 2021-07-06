#!/usr/bin/env python3

import argparse
import copy
import json
import sys
import tempfile
import tqdm
import warnings
import zipfile
from os.path import join

import numpy as np

import rasterio as rio
from rasterio.windows import Window

import torch
import torch.hub


def torch_hub_load_local(hubconf_dir: str, entrypoint: str, *args, **kwargs):
    """Same as torch.hub.load(), minus the downloading part.
    Args:
        hubconf_dir (str): A directory containing a hubconf.py file.
        entrypoint (str): Name of a callable present in hubconf.py.
    Returns:
        Any: The output from calling the entrypoint.
    """
    from torch.hub import (sys, import_module, MODULE_HUBCONF,
                           _load_entry_from_hubconf)

    sys.path.insert(0, hubconf_dir)

    hub_module = import_module(MODULE_HUBCONF, join(hubconf_dir,
                                                    MODULE_HUBCONF))

    entry = _load_entry_from_hubconf(hub_module, entrypoint)
    out = entry(*args, **kwargs)

    return out


def command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architectures",
                        required=False,
                        nargs='+',
                        choices=['both', 'cheaplab', 'fpn-resnet18'],
                        default=['cheaplab', 'fpn-resnet18'])
    parser.add_argument("--chunksize", required=False, type=int, default=64)
    parser.add_argument("--device", required=False, type=str, default="cuda")
    parser.add_argument("--exit-early",
                        required=False,
                        type=bool,
                        default=False)
    parser.add_argument("--infile", required=True, type=str)
    parser.add_argument("--level",
                        required=False,
                        choices=['L2A', 'L1C'],
                        default='L1C')
    parser.add_argument("--outfile-final", required=True, type=str)
    parser.add_argument("--outfile-raw", required=True, type=str)
    parser.add_argument("--preshrink", required=False, type=int, default=8)
    parser.add_argument("--stride", required=False, type=int, default=107)
    parser.add_argument("--window-size", required=False, type=int, default=256)
    return parser


if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    args = command_line_parser().parse_args()

    if args.level == 'L1C':
        channel_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    elif args.level == 'L2A':
        channel_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    num_classes = 2

    fpn_entrypoint = "make_segm_fpn_resnet"
    fpn_hubconf_dir = "pytorch-fpn"
    fpn_entrypoint_kwargs = {
        "name": "resnet18",
        "fpn_type": "fpn",
        "num_classes": 2,
        "fpn_channels": 256,
        "in_channels": len(channel_order),
        "out_size": [256, 256]
    }

    cheaplab_entrypoint = "make_cheaplab_model"
    cheaplab_hubconf_dir = "cheaplab"
    cheaplab_entrypoint_kwargs = {
        "preshrink": args.preshrink,
        "num_channels": len(channel_order),
    }

    if args.level == 'L1C':
        fpn_files = ["/workdir/models/fpn-resnet/L1C/0/train/model-bundle.zip"]
        cheaplab_files = [
            f"/workdir/models/cheaplab/L1C/{i}/train/model-bundle.zip"
            for i in [0, 1]
        ]
    elif args.level == 'L2A':
        fpn_files = [
            f"/workdir/models/fpn-resnet/L2A/{i}/train/model-bundle.zip"
            for i in [0, 1]
        ]
        cheaplab_files = [
            f"/workdir/models/cheaplab/L2A/{i}/train/model-bundle.zip"
            for i in [0, 1, 2]
        ]

    fpns = []
    for model_bundle in fpn_files:
        with zipfile.ZipFile(model_bundle, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_ref.extractall(tmpdirname)
                model = torch_hub_load_local(
                    f"{tmpdirname}/modules/{fpn_hubconf_dir}/", fpn_entrypoint,
                    **fpn_entrypoint_kwargs)
                model = model.to(args.device)
                model.load_state_dict(
                    torch.load(f"{tmpdirname}/model.pth",
                               map_location=torch.device(args.device)))
                model.eval()
                fpns.append(model)

    cheaplabs = []
    for model_bundle in cheaplab_files:
        with zipfile.ZipFile(model_bundle, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_ref.extractall(tmpdirname)
                model = torch_hub_load_local(
                    f"{tmpdirname}/modules/{cheaplab_hubconf_dir}/",
                    cheaplab_entrypoint, **cheaplab_entrypoint_kwargs)
                model = model.to(args.device)
                model.load_state_dict(
                    torch.load(f"{tmpdirname}/model.pth",
                               map_location=torch.device(args.device)))
                model.eval()
                cheaplabs.append(model)

    models = []
    if 'cheaplab' in args.architectures or 'both' in args.architectures:
        models += cheaplabs
    if 'fpn-resnet18' in args.architectures or 'both' in args.architectures:
        models += fpns

    if args.exit_early:
        sys.exit()

    infile = args.infile

    with rio.open(infile, 'r') as infile_ds, torch.no_grad():
        out_final_profile = copy.deepcopy(infile_ds.profile)
        out_final_profile.update({
            'compress': 'lzw',
            'dtype': np.uint8,
            'count': 1,
            'bigtiff': True,
        })
        out_raw_profile = copy.deepcopy(infile_ds.profile)
        out_raw_profile.update({
            'compress': 'lzw',
            'dtype': np.float32,
            'count': 1,
            'bigtiff': True,
        })
        width = infile_ds.width
        height = infile_ds.height
        ar_out = torch.zeros((num_classes, height, width), dtype=torch.float32)
        pixel_hits = torch.zeros((num_classes, height, width),
                                 dtype=torch.uint8)

        batches = []
        for i in range(0, width, args.stride):
            for j in range(0, height, args.stride):
                batches.append((i, j))
        batches = [
            batches[i:i + args.chunksize]
            for i in range(0, len(batches), args.chunksize)
        ]

        for batch in tqdm.tqdm(batches):
            ijs = []
            windows = []

            for (i, j) in batch:
                window = [
                    infile_ds.read(ch + 1,
                                   window=Window(i, j, args.window_size,
                                                 args.window_size)).astype(
                                                     np.float32)
                    for ch in channel_order
                ]
                window = np.stack(window, axis=0)
                if window.shape[-1] != args.window_size or window.shape[
                        -2] != args.window_size:
                    continue
                windows.append(window)
                ijs.append((i, j))

            if len(windows) > 0:
                windows = np.stack(windows, axis=0)
                windows = torch.from_numpy(windows).to(dtype=torch.float32,
                                                       device=args.device)
                raws = [model(windows) for model in models]
                probs = [raw.softmax(dim=1) for raw in raws]
                prob = sum(probs) / len(probs)
                prob = prob.cpu()
                for k, (i, j) in enumerate(ijs):
                    ar_out[:, j:(j + args.window_size),
                           i:(i + args.window_size)] += prob[k, ...]  # sic
                    pixel_hits[:, j:(j + args.window_size),
                               i:(i + args.window_size)] += 1

    ar_out = ar_out.numpy()
    pixel_hits = pixel_hits.numpy()
    ar_out /= pixel_hits

    outfile_raw = args.outfile_raw
    outfile_final = args.outfile_final

    with rio.open(outfile_raw, 'w', **out_raw_profile) as outfile_raw_ds, \
         rio.open(outfile_final, 'w', **out_final_profile) as outfile_final_ds, \
         rio.open(infile, 'r') as infile_ds:
        ar_out *= (infile_ds.read(1) != 0)
        fg = (ar_out[1] > ar_out[0]).astype(np.uint8)
        outfile_raw_ds.write(ar_out[1],
                             indexes=1,
                             window=Window(0, 0, width, height))
        outfile_final_ds.write(fg,
                               indexes=1,
                               window=Window(0, 0, width, height))
