# frozen_string_literal: true

require_relative "lib/shoulder_segmenter/version"

Gem::Specification.new do |spec|
  spec.name        = "shoulder_segmenter"
  spec.version     = ShoulderSegmenter::VERSION
  spec.authors     = ["Jonathan Siegel"]
  spec.email       = ["jonathan@siegel.io"]

  spec.summary     = "Ruby segmentation of bones from CT volumes via libtorch"
  spec.description = "Pure-Ruby orchestration around a TorchScript-traced nnU-Net (bones task) " \
                     "from TotalSegmentator. Loads CT volumes, preprocesses, runs sliding-window " \
                     "inference via torch-rb, and emits per-voxel bone label maps."
  spec.homepage    = "https://github.com/aluminumio/shoulder_segmenter"
  spec.license     = "MIT"

  spec.required_ruby_version = ">= 3.3.0"

  spec.metadata["homepage_uri"]    = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["bug_tracker_uri"] = "#{spec.homepage}/issues"

  spec.files = Dir[
    "lib/**/*.rb",
    "script/golden/network_config.yaml",
    "README.md",
    "LICENSE",
    "shoulder_segmenter.gemspec"
  ]
  spec.require_paths = ["lib"]

  spec.add_dependency "numo-narray", "~> 0.9"
  spec.add_dependency "torch-rb",    "~> 0.20"
  spec.add_dependency "nifti-ruby", ">= 0"
  spec.add_dependency "dicom_seg",  ">= 0"

  spec.add_development_dependency "bundler", ">= 2.0"
  spec.add_development_dependency "rspec",   "~> 3.13"
end
