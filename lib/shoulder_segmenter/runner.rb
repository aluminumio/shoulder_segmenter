# frozen_string_literal: true

module ShoulderSegmenter
  # Orchestrates the end-to-end pipeline:
  #   Nifti::Volume → preprocess → sliding-window inference → LabelMap
  #
  # Phase 2 scope: no spacing resample (we assume fixture is already at target
  # spacing); no TTA. Both noted in INTRODUCTION.md and README. Plumbed so a
  # Phase 3 patch can drop them in without touching the public API.
  class Runner
    attr_reader :model

    def initialize(model:)
      @model = model
    end

    def call(volume)
      arr        = Preprocess.to_narray(volume)
      normalized = Preprocess.ct_normalize(arr, model.config.normalization)
      labels     = SlidingWindow.run(normalized, model: model)

      LabelMap.new(
        data:       labels,
        labels:     model.config.labels,
        affine:     (volume.respond_to?(:affine) ? volume.affine : nil),
        voxel_size: (volume.respond_to?(:voxel_size) ? volume.voxel_size : nil)
      )
    end
  end
end
