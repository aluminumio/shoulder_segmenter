# frozen_string_literal: true

require "numo/narray"

module ShoulderSegmenter
  # Volume preprocessing: read raw bytes from a Nifti::Volume, convert to a
  # Numo::SFloat [D, H, W] array, resample to target spacing, intensity-normalize.
  #
  # nnU-Net's "CTNormalization" for the bones task is a clamp + z-score using
  # dataset-wide statistics persisted into `network_config.yaml`.
  #
  # NOTE Phase 2: resampling is a TODO — for the canonical-patch (Tier A) spec we
  # bypass it. Tier B end-to-end uses a fixture already authored at target spacing
  # so the identity path is correct.
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

      # NIfTI is Fortran-order (x fastest). Reshape into [D, H, W] = [z, y, x].
      sx, sy, sz = shape
      arr.reshape(sz, sy, sx)
    end

    # CT normalization per nnU-Net: clamp to [percentile_00_5, percentile_99_5]
    # then z-score with dataset (mean, std). Values come from network_config.yaml.
    def ct_normalize(arr, normalization)
      lo   = normalization.fetch("clip_min").to_f
      hi   = normalization.fetch("clip_max").to_f
      mean = normalization.fetch("mean").to_f
      std  = normalization.fetch("std").to_f

      out = arr.clip(lo, hi)
      ((out - mean) / std).cast_to(Numo::SFloat)
    end

    def unpack_float(bytes, fmt, shape)
      n = shape.inject(1, :*)
      Numo::SFloat.cast(bytes.unpack("#{fmt}#{n}"))
    end
  end
end
