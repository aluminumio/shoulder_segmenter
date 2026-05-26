# frozen_string_literal: true

require "numo/narray"
require "nifti"
require "tmpdir"

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

  describe "#to_nifti" do
    # Distinct values per voxel so we'd catch any axis-ordering bug.
    let(:data) do
      Numo::UInt8[
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
      ]
    end

    it "writes a label-map NIfTI that round-trips through Nifti.load" do
      Dir.mktmpdir do |dir|
        path = File.join(dir, "labels.nii.gz")
        lmap.to_nifti(path)

        round = Nifti.load(path)
        expect(round.shape).to eq([2, 2, 2])
        expect(round.dtype).to eq(:uint8)
        # Round-trip: data is [D,H,W], we transpose to [W,H,X] before write,
        # so the on-disk x-first order matches the original C-order layout
        # voxel-for-voxel.
        expect(round.to_a).to eq(data.flatten.to_a)
        expect(round.header[:intent_code]).to eq(described_class::NIFTI_INTENT_LABEL)
        expect(round.header[:intent_name]).to start_with("label_map")
      end
    end

    it "writes shape in (x, y, z) order matching the affine's axes" do
      Dir.mktmpdir do |dir|
        # Asymmetric shape proves axis order is preserved end-to-end.
        asym  = Numo::UInt8.new(2, 3, 4).seq
        lm    = described_class.new(data: asym, labels: labels,
                                    voxel_size: [0.5, 1.0, 2.0])  # (x, y, z)
        lm.to_nifti(File.join(dir, "labels.nii.gz"))

        round = Nifti.load(File.join(dir, "labels.nii.gz"))
        # asym is [D=2, H=3, W=4] in (z, y, x); on disk should be [W=4, H=3, D=2].
        expect(round.shape).to eq([4, 3, 2])
        expect(round.voxel_size).to eq([0.5, 1.0, 2.0])
      end
    end

    it "uses the provided affine when one is set" do
      Dir.mktmpdir do |dir|
        path = File.join(dir, "labels.nii.gz")
        affine = [
          [1.5, 0.0, 0.0, 10.0],
          [0.0, 1.5, 0.0, 20.0],
          [0.0, 0.0, 1.5, 30.0],
          [0.0, 0.0, 0.0,  1.0]
        ]
        described_class.new(data: data, labels: labels, affine: affine).to_nifti(path)

        round = Nifti.load(path)
        expect(round.voxel_size).to eq([1.5, 1.5, 1.5])
      end
    end

    it "falls back to a voxel-size-scaled identity affine when none is provided" do
      Dir.mktmpdir do |dir|
        path = File.join(dir, "labels.nii.gz")
        described_class.new(data: data, labels: labels, voxel_size: [2.0, 3.0, 4.0]).to_nifti(path)

        round = Nifti.load(path)
        expect(round.voxel_size).to eq([2.0, 3.0, 4.0])
      end
    end
  end
end
