# frozen_string_literal: true

require "numo/narray"

RSpec.describe ShoulderSegmenter::Model do
  # ---------------------------------------------------------------------------
  # Tier A: bit-equivalence of the inner CNN.
  #
  # Reads the canonical input patch and Python-reference output produced by
  # `script/export_totalsegmentator.py`, runs the Ruby mirror, and asserts the
  # output matches within 1e-5 absolute tolerance. Skips if the golden
  # artifacts haven't been generated yet (clean CI without Python deps).
  # ---------------------------------------------------------------------------
  context "Tier A: canonical patch round-trip" do
    before do
      skip "run script/export_totalsegmentator.py first" unless golden_artifacts_available?
    end

    let(:config) { ShoulderSegmenter::Config.load(golden("network_config.yaml")) }
    let(:model)  { described_class.load(model_weights_path, config: config) }

    let(:input_patch) do
      bytes = File.binread(golden("sample_patch_in.bin"))
      n = config.patch_size.inject(:*)
      flat = bytes.unpack("e#{n}")
      Numo::SFloat.cast(flat).reshape(*config.patch_size)
    end

    let(:expected_logits) do
      bytes = File.binread(golden("sample_patch_out.bin"))
      n = config.num_classes * config.patch_size.inject(:*)
      flat = bytes.unpack("e#{n}")
      Numo::SFloat.cast(flat).reshape(config.num_classes, *config.patch_size)
    end

    it "loads the state_dict into a parameterized Ruby module" do
      expect(model.state_dict.keys).to contain_exactly(
        "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"
      )
    end

    it "reproduces the Python reference logits within 1e-5" do
      require "torch"

      out_tensor = model.forward(input_patch)         # [1, K, D, H, W]
      out_numo   = out_tensor.squeeze(0).numo         # [K, D, H, W]

      expect(out_numo.shape).to eq(expected_logits.shape)

      diff = (out_numo - expected_logits).abs
      max_abs = diff.max
      mean_abs = diff.mean
      expect(max_abs).to be < 1e-5, "max abs diff=#{max_abs}, mean=#{mean_abs}"
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
