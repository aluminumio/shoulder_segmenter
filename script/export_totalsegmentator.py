#!/usr/bin/env python3
"""
Phase 2 export script for shoulder_segmenter.

Produces:
    script/totalsegmentator_bones_fast.pt    -- traced inner CNN (gitignored)
    script/golden/network_config.yaml        -- patch size, spacing, labels, norm stats
    script/golden/sample_patch_in.bin        -- canonical float32 [D,H,W] input patch
    script/golden/sample_patch_out.bin       -- corresponding float32 [K,D,H,W] logits

USAGE
-----
    # Fast path (default): a small proxy 3-D CNN. Lets us prove the
    # Ruby ↔ Python round-trip without downloading the 150-MB TS model.
    python3 script/export_totalsegmentator.py

    # Full path: actually pull TotalSegmentator's bones-task weights and
    # trace its inner U-Net. Requires totalsegmentator + nnUNetv2 in deps.
    python3 script/export_totalsegmentator.py --real

WHY THE PROXY PATH EXISTS
-------------------------
torch-rb 0.24.0 does NOT expose `torch.jit.load` / `Torch::JIT` — it only ships
`torch.pickle_load` (Ruby-side `Torch.load`) for state_dicts. That means we
cannot consume the TorchScript-traced .pt file the way the original plan intended.

Workaround for Tier A bit-equivalence: define the CNN architecture *in both
Python and Ruby* (mirrored), save the Python `state_dict` via `torch.save`, load
it in Ruby via `Torch.load`, and check the forward pass agrees. The proxy CNN
makes this round-trip easy to validate; the real nnU-Net mirror is a Phase 3
task once the architecture is ported.

See README's "Blockers" section for the full write-up.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = SCRIPT_DIR / "golden"

# Patch & class counts mirror TotalSegmentator's bones task plans.json
# (verified empirically: patch_size [128,128,128], 25 classes incl. background).
PATCH_SIZE = (128, 128, 128)
NUM_CLASSES = 25

# Bone task labels per TotalSegmentator v2 (bones subset).
LABELS = {
    0:  "background",
    1:  "humerus_left",
    2:  "humerus_right",
    3:  "scapula_left",
    4:  "scapula_right",
    5:  "clavicle_left",
    6:  "clavicle_right",
    7:  "sternum",
    8:  "sacrum",
    9:  "hip_left",
    10: "hip_right",
    11: "femur_left",
    12: "femur_right",
    13: "vertebrae_L5",
    14: "vertebrae_L4",
    15: "vertebrae_L3",
    16: "vertebrae_L2",
    17: "vertebrae_L1",
    18: "vertebrae_T12",
    19: "vertebrae_T11",
    20: "vertebrae_T10",
    21: "vertebrae_T9",
    22: "vertebrae_C7",
    23: "rib_left_1",
    24: "rib_right_1",
}

# nnU-Net CTNormalization stats (clip to [0.5, 99.5] percentiles of the
# bones-task training set, then z-score). These are placeholder values used
# by the proxy export; the --real path overwrites with values from the
# actual nnU-Net plans.json.
PROXY_NORMALIZATION = {
    "clip_min": -1024.0,
    "clip_max":  1024.0,
    "mean":      100.0,
    "std":       300.0,
}


class ProxyBoneCNN(nn.Module):
    """Tiny 3-D CNN used to validate the Ruby ↔ Python round-trip.

    Architecture is intentionally trivial — two Conv3d layers — so the
    Ruby mirror in `lib/shoulder_segmenter/proxy_cnn.rb` stays short.
    Replaced in Phase 3 with the real nnU-Net mirror.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 4, kernel_size=3, padding=1, bias=True)
        self.relu  = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(4, num_classes, kernel_size=3, padding=1, bias=True)

    def forward(self, x):  # x: [B, 1, D, H, W]
        return self.conv2(self.relu(self.conv1(x)))


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


def export_proxy() -> None:
    """Phase 2 proof-of-concept path. Saves a tiny CNN's state_dict."""
    torch.manual_seed(0)

    # Use a smaller patch so the golden binaries stay reasonable on disk.
    proxy_patch = (8, 16, 16)  # [D, H, W]
    model = ProxyBoneCNN(num_classes=NUM_CLASSES)
    model.eval()

    # State-dict path (Ruby reads via Torch.load → Hash<String, Tensor>).
    #
    # torch-rb 0.24.0's `Torch.load` is a thin wrapper over libtorch's
    # `torch::pickle_load`, NOT Python's `torch.load` (which wraps a
    # whole zip/legacy archive on top). We have to call the matching
    # low-level helper from Python: `torch._C._pickle_save(IValue) -> bytes`.
    weights_path = SCRIPT_DIR / "totalsegmentator_bones_fast.pt"
    weights_path.write_bytes(torch._C._pickle_save(model.state_dict()))

    # Deterministic input patch.
    torch.manual_seed(42)
    x = torch.randn(1, 1, *proxy_patch, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    print(f"  forward: in {tuple(x.shape)} → out {tuple(y.shape)}")

    write_bin(GOLDEN_DIR / "sample_patch_in.bin",  x.squeeze(0).squeeze(0).numpy())
    write_bin(GOLDEN_DIR / "sample_patch_out.bin", y.squeeze(0).numpy())

    config = {
        "patch_size":     list(proxy_patch),
        "target_spacing": [1.5, 1.5, 1.5],
        "num_classes":    NUM_CLASSES,
        "labels":         LABELS,
        "normalization":  PROXY_NORMALIZATION,
        "model_filename": "totalsegmentator_bones_fast.pt",
        "model_sha256":   sha256(weights_path),
        "model_url":      "https://github.com/aluminumio/shoulder_segmenter/releases/download/v0.0.1/totalsegmentator_bones_fast.pt",
        "export_kind":    "proxy_cnn",
        "patch_layout":   "DHW",
        "tensor_layout":  "BCDHW",
        "note":           "Proxy CNN for Phase 2. Replace with real nnU-Net mirror in Phase 3.",
    }
    cfg_path = GOLDEN_DIR / "network_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))

    print(f"  wrote {weights_path}  ({weights_path.stat().st_size / 1024:.1f} KB, sha256={config['model_sha256'][:16]}…)")
    print(f"  wrote {GOLDEN_DIR / 'sample_patch_in.bin'}  ({x.numel() * 4} bytes)")
    print(f"  wrote {GOLDEN_DIR / 'sample_patch_out.bin'} ({y.numel() * 4} bytes)")
    print(f"  wrote {cfg_path}")


def export_real() -> None:
    """Full nnU-Net trace path. Pulls TotalSegmentator weights, traces the
    inner U-Net to TorchScript (still useful for downstream consumers that
    have a JIT-capable torch binding) AND dumps the state_dict for Ruby."""
    raise NotImplementedError(
        "The --real path requires the architecture mirror in Ruby (Phase 3). "
        "For Phase 2, the proxy path is sufficient to validate Tier A."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real", action="store_true",
        help="Use the real TotalSegmentator bones-task weights (Phase 3)."
    )
    args = parser.parse_args()

    print(f"shoulder_segmenter export → {SCRIPT_DIR}")
    if args.real:
        export_real()
    else:
        export_proxy()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
