# frozen_string_literal: true

require "numo/narray"
require "nifti"

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

    # NIFTI_INTENT_LABEL — labels the volume as a discrete label map so other
    # tools (3D Slicer, FreeSurfer, etc.) don't interpolate values across class
    # boundaries.
    NIFTI_INTENT_LABEL = 1002

    # @data is laid out [D, H, W] (z, y, x) to match the torch NCDHW
    # convention used during inference. NIfTI on-disk expects [x, y, z], so
    # transpose before writing. affine and voxel_size are already in (x, y, z)
    # — they were preserved from the input Nifti::Volume.
    def to_nifti(path)
      Nifti.write(
        data.transpose(2, 1, 0).dup,
        path,
        affine:      affine || default_affine,
        voxel_size:  voxel_size,
        intent_code: NIFTI_INTENT_LABEL,
        intent_name: "label_map"
      )
    end

    private

    # Identity affine scaled by voxel_size when no affine was provided
    # (consumer didn't carry physical-space info through the pipeline).
    def default_affine
      sx, sy, sz = voxel_size || [1.0, 1.0, 1.0]
      [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    end

    def resolve_id(label)
      case label
      when Integer then label
      when String, Symbol then label_id(label.to_s)
      else raise ArgumentError, "label must be an Integer id or a String/Symbol name"
      end
    end
  end
end
