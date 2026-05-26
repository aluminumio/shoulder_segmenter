# frozen_string_literal: true

require "numo/narray"

RSpec.describe ShoulderSegmenter::Model do
  # ---------------------------------------------------------------------------
  # Tier A: bit-equivalence of the inner CNN.
  #
  # Reads the canonical input patch and Python-reference output produced by
  # `script/export_totalsegmentator.py --inspect`, runs the Ruby mirror, and
  # asserts the output matches Python within numerical tolerance plus exact
  # argmax agreement (the meaningful signal for downstream label maps).
  #
  # Tolerance note: with the real PlainConvUNet, max abs logit diff is ~2e-3,
  # mean ~3e-5. Both Ruby and Python run on the same libtorch 2.12 ABI, so the
  # residual difference is fp32 accumulation-order nondeterminism in the conv
  # backend (MKL/MKLDNN threading), not a structural mismatch. The argmax
  # match is the load-bearing assertion.
  # ---------------------------------------------------------------------------
  context "Tier A: canonical patch round-trip" do
    before do
      skip "run script/export_totalsegmentator.py --inspect first" \
        unless golden_artifacts_available?
    end

    let(:config) { ShoulderSegmenter::Config.load(golden("network_config.yaml")) }
    let(:model)  { described_class.load(model_weights_path, config: config) }
    let(:patch)  { config.golden_patch }

    let(:input_patch) do
      bytes = File.binread(golden("sample_patch_in.bin"))
      n = patch.inject(:*)
      flat = bytes.unpack("e#{n}")
      Numo::SFloat.cast(flat).reshape(*patch)
    end

    let(:expected_logits) do
      bytes = File.binread(golden("sample_patch_out.bin"))
      n = config.num_classes * patch.inject(:*)
      flat = bytes.unpack("e#{n}")
      Numo::SFloat.cast(flat).reshape(config.num_classes, *patch)
    end

    it "loads the full nnU-Net state_dict into the Ruby mirror" do
      # Canonical parameter count depends on n_stages:
      #   5 stages: 5*2*4 + 4*2*4 + 4*2 + 4*2 = 40+32+8+8 = 88
      #   6 stages: 6*2*4 + 5*2*4 + 5*2 + 5*2 = 48+40+10+10 = 108
      arch_yaml = YAML.safe_load_file(golden("nnunet_architecture.yaml"))
      n_stages = arch_yaml.fetch("n_stages")
      expected_keys = n_stages * 2 * 4 + (n_stages - 1) * 2 * 4 + (n_stages - 1) * 2 * 2
      expect(model.state_dict.size).to eq(expected_keys)
      expect(model.state_dict.keys).to include(
        "encoder.stages.0.0.convs.0.conv.weight",
        "encoder.stages.#{n_stages - 1}.0.convs.1.norm.bias",
        "decoder.transpconvs.0.weight",
        "decoder.seg_layers.#{n_stages - 2}.weight"
      )
    end

    it "reproduces the Python reference logits (argmax exact, logits within numerical noise)" do
      require "torch"

      out_tensor = model.forward(input_patch)         # [1, K, D, H, W]
      out_numo   = out_tensor.squeeze(0).numo         # [K, D, H, W]

      expect(out_numo.shape).to eq(expected_logits.shape)

      diff = (out_numo - expected_logits).abs
      max_abs = diff.max
      mean_abs = diff.mean

      # Argmax labels must match exactly — this is the only thing the
      # downstream sliding-window/label-map pipeline depends on.
      our_argmax = out_numo.max_index(axis: 0) % config.num_classes
      exp_argmax = expected_logits.max_index(axis: 0) % config.num_classes
      mismatch = our_argmax.ne(exp_argmax).count_true
      expect(mismatch).to eq(0), "argmax mismatch at #{mismatch} voxels (max abs=#{max_abs}, mean=#{mean_abs})"

      # Logit tolerance: real model accumulates fp32 over many conv ops; the
      # ceiling here is set to swallow MKL threading nondeterminism but still
      # catch a structural drift (~1e-1+) immediately.
      expect(max_abs).to be < 5e-3, "max abs diff=#{max_abs}, mean=#{mean_abs}"
    end
  end

  # ---------------------------------------------------------------------------
  # Sanity: every spec gets a consistent surface, even when the heavy artifacts
  # are missing.
  # ---------------------------------------------------------------------------
  it "raises ModelLoadError when weights are missing" do
    expect {
      described_class.load("/tmp/does_not_exist.pt", config: instance_double(ShoulderSegmenter::Config, model_url: nil))
    }.to raise_error(ShoulderSegmenter::ModelLoadError, /missing weights/)
  end
end
