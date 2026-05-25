# frozen_string_literal: true

require "numo/narray"

RSpec.describe ShoulderSegmenter::LabelMap do
  let(:labels) { { 0 => "background", 1 => "humerus_left", 2 => "scapula_left" } }
  let(:data)   { Numo::UInt8[[[0, 1], [2, 0]], [[1, 1], [0, 2]]] }

  subject(:lmap) { described_class.new(data: data, labels: labels) }

  it "exposes shape and labels" do
    expect(lmap.shape).to eq([2, 2, 2])
    expect(lmap.label_name(1)).to eq("humerus_left")
    expect(lmap.label_id("scapula_left")).to eq(2)
  end

  it "produces a per-label boolean mask" do
    mask = lmap.mask_for(:humerus_left)
    expect(mask.count_true).to eq(3)
  end

  it "looks up the dominant label at a coordinate" do
    expect(lmap.dominant_label_at(0, 0, 1)).to eq(1)
    expect(lmap.dominant_label_at(1, 1, 1)).to eq(2)
  end

  it "rejects non-3D arrays" do
    expect { described_class.new(data: Numo::UInt8[1, 2, 3], labels: labels) }
      .to raise_error(ArgumentError, /3-D/)
  end
end
