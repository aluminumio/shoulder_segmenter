# frozen_string_literal: true

require "numo/narray"
require "nifti"

RSpec.describe ShoulderSegmenter::Runner do
  before do
    skip "run script/export_totalsegmentator.py first" unless golden_artifacts_available?
    skip "nifti-ruby missing" unless defined?(Nifti)
  end

  # Phase 2 Tier B is *deferred*: end-to-end bit-equivalence against a Python
  # TotalSegmentator reference requires the real nnU-Net mirror (Phase 3).
  # Until then, the most we can prove here is that the orchestration runs
  # end-to-end on a NIfTI fixture without exploding, and that the output
  # geometry matches the input.
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
