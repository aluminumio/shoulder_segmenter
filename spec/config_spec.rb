# frozen_string_literal: true

RSpec.describe ShoulderSegmenter::Config do
  before do
    skip "run script/export_totalsegmentator.py first" unless File.exist?(golden("network_config.yaml"))
  end

  let(:config) { described_class.load(golden("network_config.yaml")) }

  it "parses patch_size as a 3-element list" do
    expect(config.patch_size.length).to eq(3)
    expect(config.patch_size).to all(be_an(Integer))
  end

  it "exposes the 25-class bone label dict" do
    expect(config.num_classes).to eq(25)
    expect(config.labels[0]).to eq("background")
    expect(config.labels[1]).to eq("humerus_left")
    expect(config.labels[2]).to eq("humerus_right")
  end

  it "carries the SHA256 of the weights blob" do
    expect(config.model_sha256).to match(/\A[0-9a-f]{64}\z/)
  end
end
