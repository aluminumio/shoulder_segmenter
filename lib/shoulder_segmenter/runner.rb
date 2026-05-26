# frozen_string_literal: true

module ShoulderSegmenter
  # Orchestrates the end-to-end pipeline:
  #   Nifti::Volume → preprocess (normalize + resample) → sliding-window
  #   inference → resample-back → LabelMap
  #
  # Spacing handling: we resample from the input volume's native voxel size to
  # the network's training spacing (3 mm isotropic for total-fast), run the
  # sliding window at that scale, then nearest-neighbour-resample the label
  # volume back to the input shape so callers can mask the original CT directly.
  class Runner
    attr_reader :model

    def initialize(model:)
      @model = model
    end

    def call(volume)
      native_arr      = Preprocess.to_narray(volume)
      native_spacing  = voxel_size_dhw(volume)
      target_spacing  = model.config.target_spacing

      resampled = if native_spacing
                    Preprocess.resample_to_spacing(native_arr,
                                                   current_spacing: native_spacing,
                                                   target_spacing:  target_spacing)
                  else
                    # Caller didn't tell us the spacing — assume the volume is
                    # already at target spacing (true for the synthetic spec
                    # fixture; explicit in README for production callers).
                    native_arr
                  end

      normalized   = Preprocess.ct_normalize(resampled, model.config.normalization)
      labels_small = SlidingWindow.run(normalized, model: model)
      labels       = Preprocess.resample_labels_to_shape(labels_small,
                                                         target_shape: native_arr.shape)

      LabelMap.new(
        data:       labels,
        labels:     model.config.labels,
        affine:     (volume.respond_to?(:affine) ? volume.affine : nil),
        voxel_size: (volume.respond_to?(:voxel_size) ? volume.voxel_size : nil)
      )
    end

    private

    # Nifti::Volume's voxel_size is in (x, y, z) but our array is (z, y, x).
    def voxel_size_dhw(volume)
      return nil unless volume.respond_to?(:voxel_size) && volume.voxel_size
      vx = Array(volume.voxel_size)
      return nil unless vx.length == 3
      vx.reverse
    end
  end
end
