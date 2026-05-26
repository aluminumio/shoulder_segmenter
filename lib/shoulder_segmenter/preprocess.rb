# frozen_string_literal: true

require "numo/narray"

module ShoulderSegmenter
  # Volume preprocessing: read raw bytes from a Nifti::Volume, convert to a
  # Numo::SFloat [D, H, W] array, resample to target spacing, intensity-normalize.
  #
  # nnU-Net's "CTNormalization" for the bones task is a clamp to the training
  # set's 0.5/99.5 percentiles followed by a z-score using dataset mean/std.
  # The percentiles + stats live in `network_config.yaml` (pulled out of the
  # nnUNetPlans by `script/export_totalsegmentator.py --inspect`).
  module Preprocess
    module_function

    # @param volume [Nifti::Volume]
    # @return [Numo::SFloat] [D, H, W] in NIfTI's (z, y, x) order
    def to_narray(volume)
      shape = volume.shape
      raise ConfigError, "expected 3-D volume, got shape=#{shape.inspect}" unless shape.length == 3

      arr =
        case volume.dtype
        when :int16   then unpack_float(volume.raw_bytes, "s<", shape)
        when :int32   then unpack_float(volume.raw_bytes, "l<", shape)
        when :uint8   then unpack_float(volume.raw_bytes, "C",  shape)
        when :float32 then unpack_float(volume.raw_bytes, "e",  shape)
        when :float64 then unpack_float(volume.raw_bytes, "E",  shape)
        else
          raise ConfigError, "unsupported dtype for preprocessing: #{volume.dtype}"
        end

      sx, sy, sz = shape
      arr.reshape(sz, sy, sx)
    end

    # nnU-Net CTNormalization: clamp to [percentile_00_5, percentile_99_5]
    # then z-score with dataset (mean, std). Values come from network_config.yaml.
    def ct_normalize(arr, normalization)
      lo   = normalization.fetch("clip_min").to_f
      hi   = normalization.fetch("clip_max").to_f
      mean = normalization.fetch("mean").to_f
      std  = normalization.fetch("std").to_f

      out = arr.clip(lo, hi)
      ((out - mean) / std).cast_to(Numo::SFloat)
    end

    # Resample a [D, H, W] volume from `current_spacing` to `target_spacing`
    # (mm) using trilinear interpolation. We piggyback on torch-rb's 5-D
    # `interpolate` so the heavy lifting stays in libtorch.
    #
    # @param arr [Numo::SFloat] [D, H, W]
    # @param current_spacing [Array<Float>] mm-per-voxel, [D, H, W] order
    # @param target_spacing  [Array<Float>] mm-per-voxel, [D, H, W] order
    # @return [Numo::SFloat] resampled [D', H', W']
    def resample_to_spacing(arr, current_spacing:, target_spacing:)
      return arr if current_spacing.zip(target_spacing).all? { |c, t| (c - t).abs < 1e-6 }

      require "torch"
      new_shape = arr.shape.zip(current_spacing, target_spacing).map do |dim, cur, tgt|
        [(dim * cur / tgt).round, 1].max
      end
      tensor = Torch.from_numo(arr).unsqueeze(0).unsqueeze(0) # [1,1,D,H,W]
      resampled = Torch::NN::Functional.interpolate(
        tensor, size: new_shape, mode: "trilinear", align_corners: false
      )
      resampled.squeeze(0).squeeze(0).numo
    end

    # Resample a label volume back from `working_spacing` to `target_spacing`
    # at a known shape (the original input's). We use nearest-neighbour so the
    # output stays an integer label map — trilinear would invent fractional
    # class IDs at boundaries.
    def resample_labels_to_shape(labels, target_shape:)
      return labels if labels.shape == target_shape

      require "torch"
      tensor = Torch.from_numo(labels.cast_to(Numo::SFloat)).unsqueeze(0).unsqueeze(0)
      resampled = Torch::NN::Functional.interpolate(
        tensor, size: target_shape, mode: "nearest"
      )
      resampled.squeeze(0).squeeze(0).numo.cast_to(labels.class)
    end

    def unpack_float(bytes, fmt, shape)
      n = shape.inject(1, :*)
      Numo::SFloat.cast(bytes.unpack("#{fmt}#{n}"))
    end
  end
end
