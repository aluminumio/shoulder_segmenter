# shoulder_segmenter

Ruby segmentation of bones from CT volumes via libtorch.

A 1:1 port (in progress) of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)'s `bones` task — the nnU-Net "fast" model that labels 24 bone classes (humerus, scapula, clavicle, sternum, vertebrae, ribs, sacrum, hip, femur) plus background — onto the [`torch-rb`](https://github.com/ankane/torch.rb) Ruby bindings to LibTorch.

## Status

**Phase 2 (this release): proof-of-concept**

- Gem skeleton, public API, sliding-window orchestration, label-map abstraction.
- Round-trip pipeline: Python `torch.save` of a state-dict → Ruby `Torch.load` → Ruby `Torch::NN::Module` mirror → forward pass.
- Ships a **proxy CNN** (two `Conv3d` layers) on both sides so the round-trip can be validated end-to-end without the 150 MB nnU-Net weights.
- Tier A bit-equivalence spec passes: max abs diff = **2.4e-7** vs the Python reference logits, on a [25, 8, 16, 16] output tensor.
- Tier B end-to-end spec exercises a synthetic 16×16×16 NIfTI fixture all the way through preprocessing → sliding-window inference → label map.

**Phase 3 (next): real nnU-Net**

- Mirror nnU-Net's actual U-Net architecture in `lib/shoulder_segmenter/nnunet.rb` so the same state-dict load path lifts the real `~/.cache/totalsegmentator` weights into Ruby.
- Hook up the `--real` branch of `script/export_totalsegmentator.py` to pull TotalSegmentator's bones-task plans + weights and dump them to the same pickle format.
- Spacing resample (currently a TODO in `Preprocess`) and reflection padding (currently zero-pad) to match nnU-Net's preprocessor.

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
seg.labels                         # => { 1 => "humerus_left", 2 => "scapula_left", ... }
seg.mask_for(:humerus_left)        # => 3-D boolean Numo::NArray
seg.dominant_label_at(z, y, x)     # => Integer class id
seg.to_nifti("out.nii.gz")         # Phase 3 — needs nifti-ruby writer
```

## How the Python ↔ Ruby model round-trip works

nnU-Net wraps a fixed-shape inner CNN with a lot of dynamic Python (per-axis resampling, sliding-window patching with Gaussian blending, optional test-time augmentation). That dynamism is not traceable, so the original plan was to `torch.jit.trace` just the inner CNN and consume the TorchScript blob from Ruby.

**Blocker:** `torch-rb 0.24.0` does NOT expose `torch::jit::load`. It only exposes `torch::pickle_load` (Ruby-side `Torch.load`) for IValue pickles.

**Workaround used here:**

1. Define the CNN architecture *in both Python and Ruby* (`script/export_totalsegmentator.py` and `lib/shoulder_segmenter/proxy_cnn.rb` — the Phase 2 proxy; `lib/shoulder_segmenter/nnunet.rb` will be the Phase 3 real one).
2. In Python: build the model, then `torch._C._pickle_save(model.state_dict())` → bytes → file. This is the libtorch low-level pickler, NOT Python's `torch.save` (which wraps it in a zip/legacy container Ruby doesn't grok).
3. In Ruby: `Torch.load(path)` returns a `Hash<String, Torch::Tensor>`. Walk it, `param.copy!(value)` into the matching `Torch::NN::Module` slots. Standard forward call from there.

The Tier A spec verifies the canonical patch's output matches the Python reference within 1e-5 (actual measured diff: ~2.4e-7).

## Specs

```sh
bundle exec rspec
```

Tier A (canonical patch) and Tier B (end-to-end NIfTI fixture) both require the artifacts from `script/export_totalsegmentator.py`:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r script/requirements.txt
python3 script/export_totalsegmentator.py        # proxy CNN; Phase 2 default
python3 script/export_totalsegmentator.py --real # Phase 3, NotImplementedError today
```

Outputs:

- `script/totalsegmentator_bones_fast.pt` — pickled state-dict (gitignored; uploaded to GH releases)
- `script/golden/network_config.yaml` — patch size, target spacing, label dict, normalization stats, model SHA256
- `script/golden/sample_patch_in.bin` — `[D,H,W]` float32 canonical input
- `script/golden/sample_patch_out.bin` — `[K,D,H,W]` float32 reference logits

If the artifacts aren't present, the heavy specs skip and the cheap ones (`LabelMap`, error paths) still run.

## Architecture map

| File | Purpose |
|---|---|
| `lib/shoulder_segmenter.rb` | Public `ShoulderSegmenter.run(volume)` entrypoint |
| `lib/shoulder_segmenter/config.rb` | Loads `network_config.yaml`: patch size, labels, norm stats |
| `lib/shoulder_segmenter/model.rb` | Loads pickled state-dict, instantiates the Ruby mirror, exposes `forward(patch)` |
| `lib/shoulder_segmenter/proxy_cnn.rb` | Phase 2 mirror: 2-layer Conv3d → ReLU → Conv3d |
| `lib/shoulder_segmenter/preprocess.rb` | NIfTI raw bytes → Numo::SFloat → CT clip+zscore (TODO: spacing resample) |
| `lib/shoulder_segmenter/sliding_window.rb` | Patch extraction + Gaussian importance-blended logits + argmax |
| `lib/shoulder_segmenter/runner.rb` | Glues preprocess → sliding window → LabelMap |
| `lib/shoulder_segmenter/label_map.rb` | Per-voxel label volume + mask/lookup helpers |

## Known gaps (Phase 3 work)

- The proxy CNN is *not* the real nnU-Net. End-to-end output is meaningless biologically; the goal of Phase 2 is just to prove the plumbing.
- `Preprocess#resample_to_target_spacing` is not implemented; fixtures must already be at the network's target spacing.
- `SlidingWindow#pad_to` does zero-padding; nnU-Net uses reflect padding.
- `LabelMap#to_nifti` raises `NotImplementedError` pending nifti-ruby's writer.
- No TTA (test-time augmentation). nnU-Net's "fast" task defaults to TTA off, so this is acceptable.
- TorchScript JIT loading from Ruby is unsupported by torch-rb. If a future torch-rb version exposes `Torch::JIT`, the architecture-mirroring approach can be retired in favor of a single `.pt` artifact.

## License

MIT — see `LICENSE`.
