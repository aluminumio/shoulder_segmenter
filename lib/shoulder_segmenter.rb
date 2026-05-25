# frozen_string_literal: true

require_relative "shoulder_segmenter/version"
require_relative "shoulder_segmenter/config"
require_relative "shoulder_segmenter/model"
require_relative "shoulder_segmenter/label_map"
require_relative "shoulder_segmenter/preprocess"
require_relative "shoulder_segmenter/sliding_window"
require_relative "shoulder_segmenter/runner"

# Ruby port of TotalSegmentator's bone-segmentation task (the nnU-Net "fast" model),
# orchestrating a TorchScript-traced inner 3D U-Net via torch-rb.
#
# Why this shape?
#   nnU-Net's full pipeline uses dynamic Python control flow (per-axis resampling,
#   sliding-window patching, per-patch Gaussian blending). That is not traceable.
#   We trace ONLY the inner fixed-shape CNN (one patch in, logits out) and re-implement
#   the orchestration in Ruby + Numo + torch-rb tensor ops.
#
# See README and `script/export_totalsegmentator.py` for how the .pt artifact is
# generated. The .pt itself is too large to commit; it is downloaded into
# `~/.cache/shoulder_segmenter/` on first use from the GitHub release.
module ShoulderSegmenter
  class Error < StandardError; end
  class ConfigError    < Error; end
  class ModelLoadError < Error; end

  # Run the bone-segmentation pipeline end-to-end on a Nifti::Volume.
  #
  # @param volume [Nifti::Volume]
  # @param model [ShoulderSegmenter::Model, nil] optional preloaded model
  # @return [ShoulderSegmenter::LabelMap]
  def self.run(volume, model: nil)
    Runner.new(model: model || Model.load_default).call(volume)
  end
end
