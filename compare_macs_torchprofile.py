#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare THOP and TorchProfile MACs for model YAML configs.

Run this script from the MotionAGFormer repository root.
Example:
python compare_macs_torchprofile.py --configs \
  configs/h36m/MotionAGFormer-base.yaml \
  configs/h36m/MotionAGFormer_DCF3.yaml \
  configs/h36m/MotionAGFormer_DCF3_CKA_student_w96.yaml
"""

import argparse
import copy
import torch
from thop import profile as thop_profile
from torchprofile import profile_macs

from utils.tools import get_config
from utils.learning import load_model


def exact_params(model):
    return sum(parameter.numel() for parameter in model.parameters())


def count_one(config_path, device):
    print("\n" + "=" * 76)
    print("Config:", config_path)

    args = get_config(config_path)
    model = load_model(args).eval().to(device)

    n_frames = int(getattr(args, "n_frames", 243))
    num_joints = int(getattr(args, "num_joints", 17))
    x = torch.randn(1, n_frames, num_joints, 3, device=device)

    params = exact_params(model)
    print("Input shape :", tuple(x.shape))
    print("Params      : {:,} ({:.4f} M)".format(params, params / 1e6))
    print("Device      :", device)

    # Run THOP on a deep copy to avoid any profiler-added buffers/hooks
    # influencing the independent TorchProfile run.
    try:
        with torch.no_grad():
            thop_macs, _ = thop_profile(
                copy.deepcopy(model).eval(),
                inputs=(x,),
                verbose=False,
            )
        thop_macs = float(thop_macs)
        print("THOP MACs       : {:.4f} G".format(thop_macs / 1e9))
    except Exception as exc:
        thop_macs = None
        print("THOP profiling failed: {}: {}".format(type(exc).__name__, exc))

    # TorchProfile traces the actual forward computation graph.
    # Keep any unsupported-operator warnings visible; they matter for auditing.
    try:
        with torch.no_grad():
            tp_macs = float(profile_macs(copy.deepcopy(model).eval(), x))
        print("TorchProfile MACs: {:.4f} G".format(tp_macs / 1e9))
    except Exception as exc:
        tp_macs = None
        print("TorchProfile profiling failed: {}: {}".format(type(exc).__name__, exc))

    if thop_macs is not None and tp_macs is not None and tp_macs > 0:
        print("TP - THOP        : {:+.4f} G".format((tp_macs - thop_macs) / 1e9))
        print("THOP / TP        : {:.2f}%".format(100.0 * thop_macs / tp_macs))

    del model, x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "config": config_path,
        "params": params,
        "thop_macs": thop_macs,
        "torchprofile_macs": tp_macs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Model config YAML paths, relative to repository root.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    cli = parser.parse_args()

    if cli.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cli.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    import torchprofile
    print("PyTorch      :", torch.__version__)
    print("TorchProfile :", getattr(torchprofile, "__version__", "version not exposed"))
    print("TorchProfile path:", torchprofile.__file__)
    print("Device       :", device)
    print("Note: do not ignore any unsupported-operator warnings during tracing.")

    results = [count_one(path, device) for path in cli.configs]

    print("\n" + "=" * 76)
    print("SUMMARY (G MACs)")
    print("{:<53s} {:>10s} {:>14s}".format("Config", "THOP", "TorchProfile"))
    for row in results:
        thop_value = "N/A" if row["thop_macs"] is None else "{:.2f}".format(row["thop_macs"] / 1e9)
        tp_value = "N/A" if row["torchprofile_macs"] is None else "{:.2f}".format(row["torchprofile_macs"] / 1e9)
        print("{:<53s} {:>10s} {:>14s}".format(row["config"][-53:], thop_value, tp_value))


if __name__ == "__main__":
    main()
