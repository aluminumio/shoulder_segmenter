#!/usr/bin/env python3
"""
Phase 3 export script for shoulder_segmenter.

Produces (per --task):
    script/<weights_filename>                -- pickled state_dict (gitignored)
    script/golden/network_config.yaml        -- patch size, spacing, labels, norm stats
    script/golden/nnunet_architecture.yaml   -- full architecture spec (Ruby mirror reads this)
    script/golden/sample_patch_in.bin        -- canonical float32 [D,H,W] input patch
    script/golden/sample_patch_out.bin       -- corresponding float32 [K,D,H,W] logits

USAGE
-----
    # Default (Phase 3.1): TotalSegmentator's part4_muscles 1.5mm sub-model
    # (task 294). This is the bones+muscles split of the full `total` model.
    # Crucially it is trained WITHOUT the femur/hip context, so a partial-FOV
    # shoulder CT cannot confuse humerus for femur (the failure mode of task 297).
    # Labels: humerus L/R, scapula L/R, clavicula L/R, femur L/R, hip L/R,
    # spinal_cord, gluteus L/R (max/med/min), autochthon L/R, iliopsoas L/R,
    # brain, skull (24 classes incl. background). Downloads ~234 MB on first run.
    python3 script/export_totalsegmentator.py --task 294

    # Legacy: total --fast (task 297, 3mm, 118 classes). Known-broken on
    # partial-FOV shoulders -- mislabels humerus as femur. Kept for backwards
    # compatibility / regression comparison.
    python3 script/export_totalsegmentator.py --task 297

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

# ----- task registry -------------------------------------------------------
#
# Each task entry describes a TotalSegmentator sub-model we know how to export.
# `dataset_name` and `trainer` are joined to build the cached-weights path:
#   ~/.totalsegmentator/nnunet/results/<dataset_name>/<trainer>__nnUNetPlans__3d_fullres/
TASKS = {
    297: {
        # "total --fast" 3mm. 118 classes, whole-body context.
        # Confidently mislabels shoulder anatomy on partial-FOV inputs because
        # the whole-body label space includes femur, which it picks for the
        # humeral head when no pelvis/spine is visible.
        "dataset_name":   "Dataset297_TotalSegmentator_total_3mm_1559subj",
        "trainer":        "nnUNetTrainer_4000epochs_NoMirroring",
        "weights_file":   "totalsegmentator_bones_fast.pt",
        "release_tag":    "v0.0.2",
        "release_asset":  "totalsegmentator_bones_fast.pt",
        "subset_labels": {
            # Bones we surface to Ruby callers via Config#bone_labels.
            0:  "background",
            25: "sacrum",
            26: "vertebrae_S1", 27: "vertebrae_L5", 28: "vertebrae_L4",
            29: "vertebrae_L3", 30: "vertebrae_L2", 31: "vertebrae_L1",
            32: "vertebrae_T12", 33: "vertebrae_T11", 34: "vertebrae_T10",
            35: "vertebrae_T9", 36: "vertebrae_T8", 37: "vertebrae_T7",
            38: "vertebrae_T6", 39: "vertebrae_T5", 40: "vertebrae_T4",
            41: "vertebrae_T3", 42: "vertebrae_T2", 43: "vertebrae_T1",
            44: "vertebrae_C7", 45: "vertebrae_C6", 46: "vertebrae_C5",
            47: "vertebrae_C4", 48: "vertebrae_C3", 49: "vertebrae_C2",
            50: "vertebrae_C1",
            69: "humerus_left", 70: "humerus_right",
            71: "scapula_left", 72: "scapula_right",
            73: "clavicula_left", 74: "clavicula_right",
            75: "femur_left", 76: "femur_right",
            77: "hip_left", 78: "hip_right",
            91: "skull",
            92: "rib_left_1", 93: "rib_left_2", 94: "rib_left_3", 95: "rib_left_4",
            96: "rib_left_5", 97: "rib_left_6", 98: "rib_left_7", 99: "rib_left_8",
            100: "rib_left_9", 101: "rib_left_10", 102: "rib_left_11", 103: "rib_left_12",
            104: "rib_right_1", 105: "rib_right_2", 106: "rib_right_3", 107: "rib_right_4",
            108: "rib_right_5", 109: "rib_right_6", 110: "rib_right_7", 111: "rib_right_8",
            112: "rib_right_9", 113: "rib_right_10", 114: "rib_right_11", 115: "rib_right_12",
            116: "sternum",
            117: "costal_cartilages",
        },
    },
    294: {
        # part4_muscles 1.5mm. 24 classes (incl background). The non-fast
        # `total` task is implemented as 5 sub-models at 1.5mm; this is the one
        # that segments humerus / scapula / clavicula. Because the label space
        # does NOT contain ribs / vertebrae / organs, an isolated shoulder CT
        # gets segmented on the basis of bone shape alone -- exactly what we
        # need for partial-FOV inputs.
        "dataset_name":   "Dataset294_TotalSegmentator_part4_muscles_1559subj",
        "trainer":        "nnUNetTrainerNoMirroring",
        "weights_file":   "totalsegmentator_part4_muscles.pt",
        "release_tag":    "v0.0.3",
        "release_asset":  "totalsegmentator_part4_muscles.pt",
        # subset_labels here is identical to the dataset labels -- every class
        # is a bone or a muscle worth surfacing. We just keep the same shape
        # so the runtime config schema is uniform across tasks.
        "subset_labels": None,  # populated from dataset.json at export time
    },
}

CONFIG_NAME = "3d_fullres"


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


def model_paths(task_id: int) -> tuple[Path, Path, Path]:
    """Return (plans.json, dataset.json, checkpoint_final.pth) under ~/.totalsegmentator."""
    from totalsegmentator.config import get_weights_dir
    t = TASKS[task_id]
    fold_dir = (Path(get_weights_dir()) / t["dataset_name"] /
                f"{t['trainer']}__nnUNetPlans__{CONFIG_NAME}")
    return (
        fold_dir / "plans.json",
        fold_dir / "dataset.json",
        fold_dir / "fold_0" / "checkpoint_final.pth",
    )


def ensure_weights_downloaded(task_id: int) -> None:
    from totalsegmentator.libs import download_pretrained_weights
    plans_path, _, ckpt_path = model_paths(task_id)
    if not ckpt_path.exists() or not plans_path.exists():
        print(f"  downloading TotalSegmentator weights for task {task_id} …")
        download_pretrained_weights(task_id)


def build_model_from_plans(task_id: int, deep_supervision: bool = False):
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans_path, ds_path, _ = model_paths(task_id)
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
    return model, conf, label_mgr, ds, plans


# --- the real export -------------------------------------------------------

def export_real(task_id: int) -> None:
    task = TASKS[task_id]
    print(f"shoulder_segmenter export task={task_id} ({task['dataset_name']}) → {SCRIPT_DIR}")
    ensure_weights_downloaded(task_id)
    plans_path, ds_path, ckpt_path = model_paths(task_id)
    plans = json.loads(plans_path.read_text())
    ds = json.loads(ds_path.read_text())

    model, conf, label_mgr, _, _ = build_model_from_plans(task_id, deep_supervision=False)
    print(f"  arch: {type(model).__name__}, {label_mgr.num_segmentation_heads} classes, "
          f"patch={conf.patch_size}, spacing={conf.spacing}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_full = ckpt["network_weights"]
    missing, unexpected = model.load_state_dict(sd_full, strict=False)
    print(f"  loaded; missing={len(missing)} unexpected={len(unexpected)} (expected 0 missing)")
    if missing:
        raise RuntimeError(f"missing keys when loading real checkpoint: {missing[:5]} (...)")
    model.eval()

    canonical_sd = {}
    for k, v in sd_full.items():
        if "all_modules" in k:
            continue
        if k.startswith("decoder.encoder."):
            continue
        canonical_sd[k] = v.detach().cpu().contiguous()
    print(f"  canonical state_dict: {len(canonical_sd)} keys "
          f"(from {len(sd_full)} raw, dropped all_modules+decoder.encoder duplicates)")

    weights_path = SCRIPT_DIR / task["weights_file"]
    weights_path.write_bytes(torch._C._pickle_save(canonical_sd))
    print(f"  wrote {weights_path}  ({weights_path.stat().st_size / 1024**2:.1f} MB, "
          f"sha256={sha256(weights_path)[:16]}…)")

    # Golden patch: divisible by 2^(n_stages-1) and deepest stage must have
    # >1 spatial element (InstanceNorm3d requirement). 32^3 works for 5 stages
    # (deepest is 2), 64^3 works for 6 stages (deepest is 2).
    n_stages = conf.network_arch_init_kwargs["n_stages"]
    min_div = 2 ** (n_stages - 1)
    golden_side = max(32, min_div * 2)
    golden_patch = (golden_side, golden_side, golden_side)
    runtime_patch = tuple(conf.patch_size)
    torch.manual_seed(42)
    x = torch.randn(1, 1, *golden_patch, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    print(f"  golden forward: in {tuple(x.shape)} → out {tuple(y.shape)}")
    write_bin(GOLDEN_DIR / "sample_patch_in.bin",  x.squeeze(0).squeeze(0).numpy())
    write_bin(GOLDEN_DIR / "sample_patch_out.bin", y.squeeze(0).numpy())

    intensity = plans["foreground_intensity_properties_per_channel"]["0"]
    arch_kwargs = conf.network_arch_init_kwargs
    arch_yaml = {
        "class_name":            conf.network_arch_class_name,
        "n_stages":              arch_kwargs["n_stages"],
        "features_per_stage":    list(arch_kwargs["features_per_stage"]),
        "kernel_sizes":          [list(k) for k in arch_kwargs["kernel_sizes"]],
        "strides":               [list(s) for s in arch_kwargs["strides"]],
        "n_conv_per_stage":      list(arch_kwargs["n_conv_per_stage"]),
        "n_conv_per_stage_decoder": list(arch_kwargs["n_conv_per_stage_decoder"]),
        "conv_bias":             arch_kwargs["conv_bias"],
        "norm_op":               arch_kwargs["norm_op"],
        "norm_op_kwargs":        dict(arch_kwargs["norm_op_kwargs"]),
        "nonlin":                arch_kwargs["nonlin"],
        "nonlin_kwargs":         dict(arch_kwargs["nonlin_kwargs"]),
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

    # Build the runtime label dict from dataset.json (id -> name).
    ds_labels = {int(v): k for k, v in ds["labels"].items()}
    subset = task["subset_labels"] or ds_labels

    release_tag = task["release_tag"]
    release_asset = task["release_asset"]
    cfg = {
        "patch_size":     list(runtime_patch),
        "golden_patch":   list(golden_patch),
        "target_spacing": list(conf.spacing),
        "num_classes":    label_mgr.num_segmentation_heads,
        "labels":         ds_labels,
        "bone_labels":    {int(k): v for k, v in subset.items()},
        "normalization":  arch_yaml["normalization"],
        "model_filename": task["weights_file"],
        "model_sha256":   sha256(weights_path),
        "model_url":      f"https://github.com/aluminumio/shoulder_segmenter/releases/download/{release_tag}/{release_asset}",
        "export_kind":    "nnunet_plain_conv_unet",
        "patch_layout":   "DHW",
        "tensor_layout":  "BCDHW",
        "task_id":        task_id,
        "trainer":        task["trainer"],
        "configuration":  CONFIG_NAME,
        "dataset_name":   task["dataset_name"],
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
                        help="Deprecated alias for --task 297. Kept for backwards compat.")
    parser.add_argument("--task", type=int, default=None,
                        help=f"TotalSegmentator task id. Known: {sorted(TASKS)}. Default: 294.")
    args = parser.parse_args()
    if args.proxy:
        export_proxy()
    else:
        task_id = args.task if args.task is not None else (297 if args.inspect else 294)
        if task_id not in TASKS:
            raise SystemExit(f"unknown task {task_id}; supported: {sorted(TASKS)}")
        export_real(task_id)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
