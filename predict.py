#!/usr/bin/env python3

import argparse
import json
import zipfile
import tempfile
import copy
import tqdm
from os.path import join

import numpy as np

import rasterio as rio
from rasterio.windows import Window

import torch
import torch.hub


# https://github.com/azavea/raster-vision/blob/2a4b834f7ac4378e8111740b931a677287208e7a/rastervision_pytorch_learner/rastervision/pytorch_learner/utils/torch_hub.py
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
    parser.add_argument("--device", required=False, type=str, default="cuda")
    parser.add_argument(
        "--entry-point-kwargs",
        required=False,
        type=str,
        default=
        '{"name": "resnet18", "fpn_type": "fpn", "num_classes": 2, "fpn_channels": 256, "in_channels": 8, "out_size": [256, 256]}'
    )
    parser.add_argument("--entry-point",
                        required=False,
                        type=str,
                        default="make_segm_fpn_resnet")
    parser.add_argument("--hubconf-dir",
                        required=False,
                        type=str,
                        default="pytorch-fpn")
    parser.add_argument("--infile", required=True, type=str)
    parser.add_argument("--input-channels",
                        required=False,
                        type=int,
                        nargs="+",
                        default=[4, 2, 1])
    parser.add_argument("--model-bundles", required=True, nargs="+", type=str)
    parser.add_argument("--outfile-final", required=True, type=str)
    parser.add_argument("--outfile-raw", required=True, type=str)
    parser.add_argument("--window-size", required=False, type=int, default=256)
    parser.add_argument("--stats-json", required=False, type=str)
    parser.add_argument("--stride", required=False, type=int, default=107)
    parser.add_argument("--chunksize", required=False, type=int, default=128)
    parser.add_argument("--num-classes", required=False, type=int, default=3)
    return parser


if __name__ == '__main__':

    args = command_line_parser().parse_args()
    args.entry_point_kwargs = json.loads(args.entry_point_kwargs)
    if 'out_size' in args.entry_point_kwargs:
        args.entry_point_kwargs['out_size'] = tuple(
            args.entry_point_kwargs['out_size'])

    if args.stats_json is not None:
        with open(args.stats_json, 'r') as f:
            stats = json.load(f)
    else:
        stats = None

    models = []
    for model_bundle in args.model_bundles:
        with zipfile.ZipFile(model_bundle, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_ref.extractall(tmpdirname)
                model = torch_hub_load_local(
                    f"{tmpdirname}/modules/{args.hubconf_dir}/",
                    args.entry_point, **args.entry_point_kwargs)
                model = model.to(args.device)
                model.load_state_dict(torch.load(f"{tmpdirname}/model.pth"))
                model.eval()
                models.append(model)

    with rio.open(args.infile, 'r') as infile_ds, torch.no_grad():
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
        ar_out = np.zeros((args.num_classes, height, width),
                          dtype=np.float32)  # sic

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
                if stats is not None:
                    window = [(infile_ds.read(ch + 1, window=Window(i, j, args.window_size,args.window_size)) - stats.get('means')[ch]).astype(np.float32) / stats.get('stds')[ch] for ch in args.input_channels]
                else:
                    window = [(infile_ds.read(ch + 1, window=Window(i, j, args.window_size,args.window_size)) - 0).astype(np.float32) / 0xff for ch in args.input_channels]
                window = np.stack(window, axis=0)
                if window.shape[-1] != args.window_size or window.shape[-2] != args.window_size:
                    continue
                windows.append(window)
                ijs.append((i, j))

            if len(windows) > 0:
                windows = np.stack(windows, axis=0)
                windows = torch.Tensor(windows).to(args.device)
                for model in models:
                    raw = model(windows)
                    raw = raw.cpu().numpy()
                    for k in range(0, len(ijs)):
                        (i, j) = ijs[k]
                        ar_out[:, j:(j + args.window_size),
                               i:(i + args.window_size)] += raw[k, ...]  # sic

    with rio.open(args.outfile_raw, 'w', **out_raw_profile) as outfile_raw_ds, \
         rio.open(args.outfile_final, 'w', **out_final_profile) as outfile_final_ds, \
         rio.open(args.infile, 'r') as infile_ds:
        ar_out *= (infile_ds.read(1) != 0)
        if args.num_classes == 3:
            fg = ((ar_out[1] > ar_out[0]) * (ar_out[1] > ar_out[2])).astype(np.uint8)
        else:
            fg = (ar_out[1] > ar_out[0]).astype(np.uint8)
        ar_out -= ar_out.max()
        ar_out = np.exp(ar_out, out=ar_out)
        ar_out = ar_out[1] / (np.sum(ar_out, axis=0))
        outfile_raw_ds.write(ar_out,
                             indexes=1,
                             window=Window(0, 0, width, height))
        outfile_final_ds.write(fg,
                               indexes=1,
                               window=Window(0, 0, width, height))
