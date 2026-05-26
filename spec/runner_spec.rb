# frozen_string_literal: true

require "numo/narray"
require "nifti"

RSpec.describe ShoulderSegmenter::Runner do
  before do
    skip "run script/export_totalsegmentator.py --inspect first" \
      unless golden_artifacts_available?
    skip "nifti-ruby missing" unless defined?(Nifti)
  end

  # Tier B: orchestration smoke test against the real model.
  #
  # We run the full pipeline (preprocess → resample → sliding window → label
  # map) on a synthetic 16x16x16 NIfTI fixture. The output is biologically
  # meaningless on synthetic noise — virtually every voxel should land on
  # the background class — but the spec verifies the geometry stays consistent
  # end-to-end and nothing along the way crashes.
  #
  # The sliding window pads the tiny fixture up to the model's patch size
  # (112x112x128), so this exercises exactly one forward pass.
  it "runs end-to-end against the NIfTI fixture from nifti-ruby" do
    nifti_fixture = File.expand_path("../../nifti-ruby/spec/fixtures/synthetic_16x16x16_uint8.nii.gz", __dir__)
    skip "missing #{nifti_fixture}" unless File.exist?(nifti_fixture)

    volume = Nifti.load(nifti_fixture)
    config = ShoulderSegmenter::Config.load(golden("network_config.yaml"))
    model  = ShoulderSegmenter::Model.load(model_weights_path, config: config)

    lmap = described_class.new(model: model).call(volume)

    expect(lmap).to be_a(ShoulderSegmenter::LabelMap)
    expect(lmap.shape).to eq([16, 16, 16])
    expect(lmap.labels.fetch(0)).to eq("background")
    expect(lmap.data.max).to be < config.num_classes
  end
end
