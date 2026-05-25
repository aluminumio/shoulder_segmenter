# frozen_string_literal: true

require "numo/narray"

module ShoulderSegmenter
  # Per-voxel label volume.
  #   data:    Numo::UInt8 of shape [D, H, W], values in [0, num_classes)
  #   labels:  Hash<Integer,String> id => human name (0 == background)
  class LabelMap
    attr_reader :data, :labels, :affine, :voxel_size

    def initialize(data:, labels:, affine: nil, voxel_size: nil)
      raise ArgumentError, "data must be 3-D Numo array" unless data.is_a?(Numo::NArray) && data.ndim == 3

      @data       = data
      @labels     = labels
      @affine     = affine
      @voxel_size = voxel_size
    end

    def shape
      data.shape
    end

    def mask_for(label)
      id = resolve_id(label)
      data.eq(id)
    end

    def dominant_label_at(z, y, x)
      data[z, y, x]
    end

    def label_name(id)
      labels[id]
    end

    def label_id(name)
      labels.invert.fetch(name.to_s)
    end

    # TODO Phase 3: real NIfTI writer (likely via nifti-ruby once writer lands).
    def to_nifti(_path)
      raise NotImplementedError, "to_nifti pending nifti-ruby writer support (Phase 3)"
    end

    private

    def resolve_id(label)
      case label
      when Integer then label
      when String, Symbol then label_id(label.to_s)
      else raise ArgumentError, "label must be an Integer id or a String/Symbol name"
      end
    end
  end
end
