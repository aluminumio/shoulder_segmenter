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

  it "exposes the TotalSegmentator total-fast label dict (118 classes)" do
    expect(config.num_classes).to eq(118)
    expect(config.labels[0]).to eq("background")
    expect(config.labels[69]).to eq("humerus_left")
    expect(config.labels[70]).to eq("humerus_right")
    expect(config.labels[116]).to eq("sternum")
  end

  it "exposes the bone-only subset map" do
    expect(config.bone_labels[0]).to eq("background")
    expect(config.bone_labels[69]).to eq("humerus_left")
    expect(config.bone_labels[71]).to eq("scapula_left")
    expect(config.bone_labels).not_to have_key(51) # heart, organ — not a bone
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
