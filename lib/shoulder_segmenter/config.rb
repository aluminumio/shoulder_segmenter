# frozen_string_literal: true

require "yaml"

module ShoulderSegmenter
  # Network and dataset configuration written out by `script/export_totalsegmentator.py`
  # alongside the traced `.pt`. We load it both for spec assertions and for
  # the runtime pipeline (patch size, target spacing, label dict, etc.).
  class Config
    DEFAULT_PATH = File.expand_path("../../script/golden/network_config.yaml", __dir__)

    attr_reader :patch_size, :golden_patch, :target_spacing, :num_classes, :labels,
                :bone_labels, :normalization, :model_filename, :model_sha256, :model_url,
                :arch_yaml_path, :raw

    def self.load(path = DEFAULT_PATH)
      raise ConfigError, "missing network config at #{path} (run script/export_totalsegmentator.py)" \
        unless File.exist?(path)

      hash = YAML.safe_load_file(path, permitted_classes: [Symbol, Float])
      arch_path = File.join(File.dirname(path), "nnunet_architecture.yaml")
      new(hash, arch_yaml_path: arch_path)
    end

    def initialize(hash, arch_yaml_path: nil)
      @raw            = hash
      @patch_size     = hash.fetch("patch_size")
      @golden_patch   = hash["golden_patch"] || hash.fetch("patch_size")
      @target_spacing = hash.fetch("target_spacing")
      @num_classes    = hash.fetch("num_classes")
      @labels         = hash.fetch("labels").transform_keys(&:to_i)
      @bone_labels    = (hash["bone_labels"] || hash["labels"]).transform_keys(&:to_i)
      @normalization  = hash.fetch("normalization")
      @model_filename = hash.fetch("model_filename")
      @model_sha256   = hash["model_sha256"]
      @model_url      = hash["model_url"]
      @arch_yaml_path = arch_yaml_path
    end

    def label_name(id)
      labels[id]
    end

    def label_id(name)
      labels.invert.fetch(name.to_s)
    end
  end
end
