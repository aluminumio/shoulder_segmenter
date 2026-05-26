# shoulder_segmenter

Ruby segmentation of bones from CT volumes via libtorch.

A 1:1 port of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)'s `total --fast` task — the nnU-Net 3 mm model that labels the full 117-class TotalSegmentator label set (humerus, scapula, clavicle, sternum, vertebrae C1..L5+S1, ribs 1..12 L/R, sacrum, hip, femur, plus all soft-tissue classes) — onto the [`torch-rb`](https://github.com/ankane/torch.rb) Ruby bindings to LibTorch. The bone-only subset is exposed as `Config#bone_labels` for downstream consumers that only care about the skeleton.

## Status

**Phase 3.1 (this release, v0.0.3): switch to part4_muscles 1.5 mm sub-model**

The `total --fast` 3 mm task (297) confidently mislabels shoulder anatomy on partial-FOV shoulder CTs — it sees a humeral head and calls it `femur_left` because the whole-body label space includes the femur and the model uses pelvic/spinal context (absent here) to disambiguate. We swap in TotalSegmentator's part4_muscles 1.5 mm sub-model (task 294), which carves out humerus/scapula/clavicula/femur/hip/spinal_cord/glutes/autochthon/iliopsoas/brain/skull (24 classes incl. background) and is trained on the higher-resolution muscle/bone split. On a partial-FOV shoulder this now produces non-zero `humerus_left/right`, `scapula_left/right`, `clavicula_left/right` voxel counts where v0.0.2 returned zero.

- Ruby `BonesUNet` is a parameter-for-parameter mirror of `dynamic_network_architectures.PlainConvUNet`, fully parameterized by `n_stages` and `features_per_stage` so it adapts to either the 5-stage 118-class task 297 or the 6-stage 24-class task 294 with no code changes.
- 108 canonical state-dict keys for task 294 (6 stages × 2 × 4 enc + 5 stages × 2 × 4 dec + 5 × 2 transpconv + 5 × 2 seg = 108) load from the real TotalSegmentator checkpoint into the Ruby module — no missing / unexpected keys, no shape mismatches.
- Tier A spec asserts 100% argmax agreement against the Python reference on a 64³ canonical patch (32³ for the 5-stage 297 net); logit-level max abs diff settles at ~2 e-3 from fp32 accumulation-order nondeterminism in the conv backend (Ruby and Python share the same libtorch 2.12 ABI).
- Tier B exercises the full pipeline (CT normalization → resample to target spacing → sliding-window inference with Gaussian importance blending and reflect padding → nearest-neighbour resample back to native voxel grid → `LabelMap`).
- Weights ship as a GitHub release asset (`v0.0.3` → `totalsegmentator_part4_muscles.pt`, 119 MB). First use downloads to `~/.cache/shoulder_segmenter/`.

**Known limitation:** the part4_muscles label space still contains `femur_left/right` and `hip_left/right`. On a partial-FOV shoulder CT the model still splits some humerus shaft into the `femur_left` class. The fix is downstream: callers that know the FOV is a shoulder should treat `humerus_left ∪ femur_left` as the humerus mask (and similarly `scapula_left ∪ hip_left`). Future revisions could either (a) crop / mask out the appendicular long-bone classes post-inference, or (b) fine-tune the model on shoulder-only data.

**Phase 2 (legacy): proof-of-concept proxy CNN**

- The 2-layer Conv3d proxy still lives in `lib/shoulder_segmenter/proxy_cnn.rb` and is selected by `network_config.yaml`'s `export_kind: proxy_cnn`. Keep it for fast unit tests where the 64-MB real model is overkill.

## Install

```sh
brew install pytorch                                  # ships libtorch 2.12.0
bundle config set build.torch-rb --with-torch-dir=/opt/homebrew/opt/pytorch
bundle install
```

Runtime dependencies: `torch-rb ~> 0.20`, `numo-narray ~> 0.9`, plus our sibling `nifti-ruby` and `dicom_seg` gems (path-deps for now, published rubygems later).

## Usage

```ruby
require "nifti"
require "shoulder_segmenter"

volume = Nifti.load("ct_chest.nii.gz")
seg    = ShoulderSegmenter.run(volume)

seg.shape                          # => [d, h, w]
seg.labels                         # => { 0 => "background", 69 => "humerus_left", ... }
seg.mask_for(:humerus_left)        # => 3-D boolean Numo::NArray
seg.dominant_label_at(z, y, x)     # => Integer class id (0..117)
```

## How the Python ↔ Ruby model round-trip works

nnU-Net wraps a fixed-shape inner CNN with a lot of dynamic Python (per-axis resampling, sliding-window patching with Gaussian blending, optional test-time augmentation). That dynamism is not traceable, so the original plan was to `torch.jit.trace` just the inner CNN and consume the TorchScript blob from Ruby.

**Blocker:** `torch-rb 0.24.0` does NOT expose `torch::jit::load`. It only exposes `torch::pickle_load` (Ruby-side `Torch.load`) for IValue pickles.

**Workaround used here:**

1. Define the CNN architecture *in both Python and Ruby* (`script/export_totalsegmentator.py` and `lib/shoulder_segmenter/bones_unet.rb`).
2. In Python: load the official TotalSegmentator checkpoint into the real `dynamic_network_architectures.PlainConvUNet`, then `torch._C._pickle_save(model.state_dict())` → bytes → file (after stripping nnU-Net's `all_modules.*` parameter aliases and the duplicated `decoder.encoder.*` shared refs, leaving 88 canonical keys).
3. In Ruby: `Torch.load(path)` returns a `Hash<String, Torch::Tensor>`. `BonesUNet#load_state_dict!` walks the dotted keys, locates the matching `Torch::NN::Parameter` slot (numeric segments index Sequential/ModuleList children; named segments are submodule attrs), and copies values in. Standard forward call from there.

The Tier A spec proves the resulting logits match Python's reference to argmax-exact, with max abs logit diff of ~2 e-3 (pure fp32 accumulation noise — both sides use the same libtorch 2.12).

## Architecture map

| File | Purpose |
|---|---|
| `lib/shoulder_segmenter.rb` | Public `ShoulderSegmenter.run(volume)` entrypoint |
| `lib/shoulder_segmenter/config.rb` | Loads `network_config.yaml`: patch size, labels, norm stats, arch YAML pointer |
| `lib/shoulder_segmenter/model.rb` | Loads pickled state-dict, instantiates `BonesUNet` (or `ProxyCNN` legacy), exposes `forward(patch)` |
| `lib/shoulder_segmenter/bones_unet.rb` | Phase 3 Ruby mirror of nnU-Net PlainConvUNet (5 encoder stages, 4 decoder stages, ConvTranspose3d wrapped from `Torch.conv_transpose3d`) |
| `lib/shoulder_segmenter/proxy_cnn.rb` | Phase 2 legacy: 2-layer Conv3d → ReLU → Conv3d |
| `lib/shoulder_segmenter/preprocess.rb` | NIfTI raw bytes → Numo::SFloat → CT clip+zscore + trilinear resample to target spacing |
| `lib/shoulder_segmenter/sliding_window.rb` | Patch extraction (reflect pad), Gaussian importance-blended logits, channel-axis argmax |
| `lib/shoulder_segmenter/runner.rb` | Glues preprocess → sliding window → label resample → LabelMap |
| `lib/shoulder_segmenter/label_map.rb` | Per-voxel label volume + mask/lookup helpers |

## Specs

```sh
bundle exec rspec
```

All heavy specs require artifacts generated by `script/export_totalsegmentator.py`:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r script/requirements.txt
python3 script/export_totalsegmentator.py --task 294  # default (Phase 3.1): part4_muscles 1.5 mm; downloads ~234 MB
python3 script/export_totalsegmentator.py --task 297  # legacy total-fast 3 mm
python3 script/export_totalsegmentator.py --proxy     # legacy 2-layer CNN
```

Outputs:

- `script/totalsegmentator_part4_muscles.pt` — pickled state-dict for task 294 (uploaded to GH releases as `v0.0.3`)
- `script/totalsegmentator_bones_fast.pt` — pickled state-dict for legacy task 297 (uploaded as `v0.0.2`)
- `script/golden/network_config.yaml` — patch size, target spacing, label dict, normalization stats, model SHA256
- `script/golden/nnunet_architecture.yaml` — full architecture spec the Ruby mirror reads
- `script/golden/sample_patch_in.bin` — canonical float32 input patch (64³ for 6-stage nets, 32³ for 5-stage)
- `script/golden/sample_patch_out.bin` — float32 Python-reference logits at the same patch shape

If the artifacts aren't present, the heavy specs skip and the cheap ones (`LabelMap`, error paths) still run.

## Known gaps

- Output classes are the full 118 — bone subsetting happens via `Config#bone_labels`. A future helper could return a `LabelMap` filtered to bones only.
- `LabelMap#to_nifti` raises `NotImplementedError` pending `nifti-ruby`'s writer.
- No TTA (test-time augmentation). nnU-Net's "fast" task defaults to TTA off, so this is acceptable.
- TorchScript JIT loading from Ruby is unsupported by torch-rb. If a future torch-rb version exposes `Torch::JIT`, the architecture-mirroring approach can be retired in favor of a single `.pt` artifact.

## License

MIT — see `LICENSE`.
