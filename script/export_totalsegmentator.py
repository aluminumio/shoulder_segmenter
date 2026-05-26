#!/usr/bin/env python3
"""
Phase 3 export script for shoulder_segmenter.

Produces:
    script/totalsegmentator_bones_fast.pt    -- pickled state_dict (gitignored)
    script/golden/network_config.yaml        -- patch size, spacing, labels, norm stats
    script/golden/nnunet_architecture.yaml   -- full architecture spec (Ruby mirror reads this)
    script/golden/sample_patch_in.bin        -- canonical float32 [D,H,W] input patch
    script/golden/sample_patch_out.bin       -- corresponding float32 [K,D,H,W] logits

USAGE
-----
    # Default (Phase 3): TotalSegmentator's `total --fast` model (task 297).
    # This is the 3mm full-CT segmentation model (PlainConvUNet, 5 stages, 118 classes
    # including all bones — humerus, scapula, clavicle, sternum, vertebrae, ribs,
    # sacrum, hip, femur). On first run the script triggers TotalSegmentator's
    # weights download (~135 MB, cached under ~/.totalsegmentator/).
    python3 script/export_totalsegmentator.py --inspect

    # Legacy proof-of-concept proxy CNN (Phase 2). Retained for fast unit tests.
    python3 script/export_totalsegmentator.py --proxy

WHY MIRROR THE ARCHITECTURE IN BOTH LANGUAGES
---------------------------------------------
torch-rb 0.24.0 does NOT expose `torch.jit.load` / `Torch::JIT` -- it only ships
`torch.pickle_load` (Ruby `Torch.load`) for state_dicts. So we cannot consume a
TorchScript-traced .pt directly. Workaround: define the architecture in Python
(this script) AND in Ruby (`lib/shoulder_segmenter/bones_unet.rb`), save Python's
state_dict via `torch._C._pickle_save`, load it in Ruby via `Torch.load`, then
copy each parameter into the matching `Torch::NN::Module` slot.

The state_dict is the contract: if any key name or shape disagrees between the
two architectures, the Ruby loader raises loudly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = SCRIPT_DIR / "golden"

# TotalSegmentator task we mirror: the "total --fast" 3mm model.
# It is the smallest single-model TotalSegmentator network that still covers
# the full bone set (humerus, scapula, clavicle, sternum, ribs, vertebrae,
# sacrum, hip, femur). The non-fast "total" task splits into five sub-models
# (one being part5_ribs) and would require five state_dict mirrors.
TASK_ID = 297
DATASET_NAME = "Dataset297_TotalSegmentator_total_3mm_1559subj"
TRAINER = "nnUNetTrainer_4000epochs_NoMirroring"
CONFIG_NAME = "3d_fullres"

# Bone labels we expose to Ruby callers. The model itself emits 118 classes
# (the full "total" label set); the LabelMap on the Ruby side projects to
# this bone-only subset by remapping the argmax volume.
BONE_LABELS = {
    0:  "background",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    91: "skull",
    92: "rib_left_1",  93: "rib_left_2",  94: "rib_left_3",  95: "rib_left_4",
    96: "rib_left_5",  97: "rib_left_6",  98: "rib_left_7",  99: "rib_left_8",
    100: "rib_left_9", 101: "rib_left_10", 102: "rib_left_11", 103: "rib_left_12",
    104: "rib_right_1",  105: "rib_right_2",  106: "rib_right_3",  107: "rib_right_4",
    108: "rib_right_5",  109: "rib_right_6",  110: "rib_right_7",  111: "rib_right_8",
    112: "rib_right_9", 113: "rib_right_10", 114: "rib_right_11", 115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages",
}


# --- helpers ---------------------------------------------------------------

def write_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    path.write_bytes(arr.tobytes())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_paths() -> tuple[Path, Path, Path]:
    """Return (plans.json, dataset.json, checkpoint_final.pth) under ~/.totalsegmentator."""
    from totalsegmentator.config import get_weights_dir
    fold_dir = (Path(get_weights_dir()) / DATASET_NAME /
                f"{TRAINER}__nnUNetPlans__{CONFIG_NAME}")
    return (
        fold_dir / "plans.json",
        fold_dir / "dataset.json",
        fold_dir / "fold_0" / "checkpoint_final.pth",
    )


def ensure_weights_downloaded() -> None:
    """Download the task 297 weights if not already cached."""
    from totalsegmentator.libs import download_pretrained_weights
    plans_path, _, ckpt_path = model_paths()
    if not ckpt_path.exists() or not plans_path.exists():
        print(f"  downloading TotalSegmentator weights for task {TASK_ID} (~135 MB)…")
        download_pretrained_weights(TASK_ID)


def build_model_from_plans(deep_supervision: bool = False):
    """Instantiate the actual nnU-Net via dynamic_network_architectures (same
    path TotalSegmentator uses at inference)."""
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans_path, ds_path, _ = model_paths()
    plans = json.loads(plans_path.read_text())
    ds = json.loads(ds_path.read_text())

    pm = PlansManager(plans)
    conf = pm.get_configuration(CONFIG_NAME)
    label_mgr = pm.get_label_manager(ds)

    model = get_network_from_plans(
        conf.network_arch_class_name,
        conf.network_arch_init_kwargs,
        conf.network_arch_init_kwargs_req_import,
        input_channels=1,
        output_channels=label_mgr.num_segmentation_heads,
        allow_init=False,
        deep_supervision=deep_supervision,
    )
    return model, conf, label_mgr, ds


# --- the real export -------------------------------------------------------

def export_real() -> None:
    print(f"shoulder_segmenter export (real nnU-Net mirror) → {SCRIPT_DIR}")
    ensure_weights_downloaded()
    plans_path, ds_path, ckpt_path = model_paths()
    plans = json.loads(plans_path.read_text())
    ds = json.loads(ds_path.read_text())

    model, conf, label_mgr, _ = build_model_from_plans(deep_supervision=False)
    print(f"  arch: {type(model).__name__}, {label_mgr.num_segmentation_heads} classes")

    # Load the official checkpoint into the model.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_full = ckpt["network_weights"]
    missing, unexpected = model.load_state_dict(sd_full, strict=False)
    # `all_modules` keys are aliases of the canonical `.conv` / `.norm` keys,
    # so a small "unexpected" list is expected. `decoder.encoder.*` are shared
    # references and also redundant.
    print(f"  loaded; missing={len(missing)} unexpected={len(unexpected)} (expected ~0 missing)")
    if missing:
        raise RuntimeError(f"missing keys when loading real checkpoint: {missing[:5]} (...)")
    model.eval()

    # Build the *canonical* state_dict the Ruby mirror consumes. We keep only
    # the unique parameter names — strip the duplicated `all_modules.*` and
    # `decoder.encoder.*` keys.
    canonical_sd = {}
    for k, v in sd_full.items():
        if "all_modules" in k:
            continue
        if k.startswith("decoder.encoder."):
            continue
        canonical_sd[k] = v.detach().cpu().contiguous()
    print(f"  canonical state_dict: {len(canonical_sd)} keys "
          f"(from {len(sd_full)} raw, dropped all_modules+decoder.encoder duplicates)")

    # Write the state_dict pickle (torch-rb 0.24.0's `Torch.load` expects this).
    weights_path = SCRIPT_DIR / "totalsegmentator_bones_fast.pt"
    weights_path.write_bytes(torch._C._pickle_save(canonical_sd))
    print(f"  wrote {weights_path}  ({weights_path.stat().st_size / 1024**2:.1f} MB, "
          f"sha256={sha256(weights_path)[:16]}…)")

    # Canonical input/output for Tier A bit-equivalence. We use a 32^3 patch
    # rather than the runtime patch (112x112x128, ~6 MB input / ~760 MB output)
    # so the golden artifacts stay small enough to ship as GH release assets.
    # The architecture is fully convolutional — any spatial dims divisible by
    # 2^(n_stages-1)=16 produce a valid forward pass.
    golden_patch = (32, 32, 32)
    runtime_patch = tuple(conf.patch_size)
    torch.manual_seed(42)
    x = torch.randn(1, 1, *golden_patch, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)  # [1, K, D, H, W]
    print(f"  golden forward: in {tuple(x.shape)} → out {tuple(y.shape)}")
    write_bin(GOLDEN_DIR / "sample_patch_in.bin",  x.squeeze(0).squeeze(0).numpy())
    write_bin(GOLDEN_DIR / "sample_patch_out.bin", y.squeeze(0).numpy())

    # Pull architecture + preprocessing knobs out of the plans for the Ruby side.
    intensity = plans["foreground_intensity_properties_per_channel"]["0"]
    arch_kwargs = conf.network_arch_init_kwargs
    arch_yaml = {
        "class_name":            conf.network_arch_class_name,
        "n_stages":              arch_kwargs["n_stages"],
        "features_per_stage":    arch_kwargs["features_per_stage"],
        "kernel_sizes":          arch_kwargs["kernel_sizes"],
        "strides":               arch_kwargs["strides"],
        "n_conv_per_stage":      arch_kwargs["n_conv_per_stage"],
        "n_conv_per_stage_decoder": arch_kwargs["n_conv_per_stage_decoder"],
        "conv_bias":             arch_kwargs["conv_bias"],
        "norm_op":               arch_kwargs["norm_op"],
        "norm_op_kwargs":        arch_kwargs["norm_op_kwargs"],
        "nonlin":                arch_kwargs["nonlin"],
        "nonlin_kwargs":         arch_kwargs["nonlin_kwargs"],
        "input_channels":        1,
        "output_channels":       label_mgr.num_segmentation_heads,
        "patch_size":            list(runtime_patch),
        "golden_patch":          list(golden_patch),
        "target_spacing":        list(conf.spacing),
        "normalization": {
            "scheme":     "CTNormalization",
            "clip_min":   float(intensity["percentile_00_5"]),
            "clip_max":   float(intensity["percentile_99_5"]),
            "mean":       float(intensity["mean"]),
            "std":        float(intensity["std"]),
        },
    }
    arch_yaml_path = GOLDEN_DIR / "nnunet_architecture.yaml"
    arch_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    arch_yaml_path.write_text(yaml.safe_dump(arch_yaml, sort_keys=False))
    print(f"  wrote {arch_yaml_path}")

    # network_config.yaml — the runtime-facing config (label map, model URL, etc.).
    cfg = {
        "patch_size":     list(runtime_patch),
        "golden_patch":   list(golden_patch),
        "target_spacing": list(conf.spacing),
        "num_classes":    label_mgr.num_segmentation_heads,
        "labels":         {int(v): k for k, v in ds["labels"].items()},
        "bone_labels":    BONE_LABELS,
        "normalization":  arch_yaml["normalization"],
        "model_filename": "totalsegmentator_bones_fast.pt",
        "model_sha256":   sha256(weights_path),
        "model_url":      "https://github.com/aluminumio/shoulder_segmenter/releases/download/v0.0.2/totalsegmentator_bones_fast.pt",
        "export_kind":    "nnunet_plain_conv_unet",
        "patch_layout":   "DHW",
        "tensor_layout":  "BCDHW",
        "task_id":        TASK_ID,
        "trainer":        TRAINER,
        "configuration":  CONFIG_NAME,
        "dataset_name":   DATASET_NAME,
    }
    cfg_path = GOLDEN_DIR / "network_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"  wrote {cfg_path}")
    print(f"  wrote sample_patch_in.bin  ({x.numel() * 4} bytes)")
    print(f"  wrote sample_patch_out.bin ({y.numel() * 4} bytes)")


# --- proxy path (kept for fast tests) --------------------------------------

class ProxyBoneCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 4, kernel_size=3, padding=1, bias=True)
        self.relu  = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(4, num_classes, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


def export_proxy() -> None:
    """Tiny 2-layer CNN used by the legacy Phase 2 fast unit tests."""
    print(f"shoulder_segmenter export (proxy CNN) → {SCRIPT_DIR}")
    torch.manual_seed(0)
    proxy_patch = (8, 16, 16)
    num_classes = 25
    model = ProxyBoneCNN(num_classes=num_classes)
    model.eval()

    weights_path = SCRIPT_DIR / "totalsegmentator_bones_fast.pt"
    weights_path.write_bytes(torch._C._pickle_save(model.state_dict()))

    torch.manual_seed(42)
    x = torch.randn(1, 1, *proxy_patch, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    write_bin(GOLDEN_DIR / "sample_patch_in.bin",  x.squeeze(0).squeeze(0).numpy())
    write_bin(GOLDEN_DIR / "sample_patch_out.bin", y.squeeze(0).numpy())

    cfg = {
        "patch_size":     list(proxy_patch),
        "target_spacing": [1.5, 1.5, 1.5],
        "num_classes":    num_classes,
        "labels":         {i: f"class_{i}" for i in range(num_classes)},
        "normalization":  {"clip_min": -1024.0, "clip_max": 1024.0, "mean": 100.0, "std": 300.0},
        "model_filename": "totalsegmentator_bones_fast.pt",
        "model_sha256":   sha256(weights_path),
        "model_url":      None,
        "export_kind":    "proxy_cnn",
        "patch_layout":   "DHW",
        "tensor_layout":  "BCDHW",
    }
    (GOLDEN_DIR / "network_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"  wrote proxy weights ({weights_path.stat().st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", action="store_true",
                        help="Use the legacy tiny proxy CNN (Phase 2; fast unit tests).")
    parser.add_argument("--inspect", action="store_true",
                        help="(default) Inspect the real TotalSegmentator total-fast task.")
    args = parser.parse_args()
    if args.proxy:
        export_proxy()
    else:
        export_real()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
