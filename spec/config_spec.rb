# frozen_string_literal: true

RSpec.describe ShoulderSegmenter::Config do
  before do
    skip "run script/export_totalsegmentator.py --inspect first" \
      unless File.exist?(golden("network_config.yaml"))
  end

  let(:config) { described_class.load(golden("network_config.yaml")) }

  it "parses patch_size as a 3-element list" do
    expect(config.patch_size.length).to eq(3)
    expect(config.patch_size).to all(be_an(Integer))
  end

  it "exposes the TotalSegmentator label dict (humerus, scapula, clavicula present)" do
    # Default task is 294 (part4_muscles 1.5mm) -- 24 classes incl background.
    # Task 297 (legacy total-fast 3mm) yields 118 classes. Both must surface
    # the shoulder bones we care about.
    expect(config.num_classes).to be >= 24
    expect(config.labels[0]).to eq("background")
    expect(config.labels.values).to include("humerus_left", "humerus_right",
                                            "scapula_left", "scapula_right",
                                            "clavicula_left", "clavicula_right")
  end

  it "exposes the bone-only subset map" do
    expect(config.bone_labels[0]).to eq("background")
    expect(config.bone_labels.values).to include("humerus_left", "scapula_left",
                                                 "clavicula_left")
  end

  it "carries the SHA256 of the weights blob" do
    expect(config.model_sha256).to match(/\A[0-9a-f]{64}\z/)
  end

  it "exposes nnU-Net CT normalization stats" do
    expect(config.normalization.fetch("clip_min")).to be < 0
    expect(config.normalization.fetch("clip_max")).to be > 0
    expect(config.normalization.fetch("std")).to be > 0
  end

  it "points at a Phase 3 architecture YAML next to it" do
    expect(config.arch_yaml_path).to end_with("nnunet_architecture.yaml")
  end
end
