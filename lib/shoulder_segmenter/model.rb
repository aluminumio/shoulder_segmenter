# frozen_string_literal: true

require_relative "proxy_cnn"
require_relative "bones_unet"

module ShoulderSegmenter
  # Wrapper around the inner CNN. Phase 3 ships the real TotalSegmentator
  # `total --fast` (task 297) nnU-Net — `dynamic_network_architectures`'s
  # PlainConvUNet — defined in both Python (`script/export_totalsegmentator.py`)
  # and Ruby (`bones_unet.rb`), with weights round-tripped via libtorch's
  # `torch::pickle_save` / `torch::pickle_load` (Ruby `Torch.load`).
  #
  # Input  shape: [B=1, C=1, D, H, W] float32 (D/H/W = config.patch_size)
  # Output shape: [B=1, K, D, H, W]  float32 logits (K = config.num_classes)
  #
  # The legacy 2-layer proxy CNN (Phase 2) is still wired up behind
  # `export_kind: proxy_cnn` for fast unit tests.
  class Model
    CACHE_DIR = File.expand_path("~/.cache/shoulder_segmenter")

    attr_reader :net, :config, :path, :state_dict

    def self.load_default(config: Config.load)
      load(default_path(config), config: config)
    end

    def self.default_path(config)
      ENV["SHOULDER_SEGMENTER_MODEL"] || begin
        local = File.expand_path("../../script/#{config.model_filename}", __dir__)
        return local if File.exist?(local)

        File.join(CACHE_DIR, config.model_filename)
      end
    end

    def self.load(path, config: Config.load)
      raise ModelLoadError, "missing weights at #{path}. " \
                            "Download from #{config.model_url || 'GH release'} or run script/export_totalsegmentator.py" \
        unless File.exist?(path)

      require "torch"
      state_dict = Torch.load(path)
      unless state_dict.is_a?(Hash)
        raise ModelLoadError, "expected Hash state_dict at #{path}, got #{state_dict.class}"
      end

      net = build_network(config).tap { |n| n.load_state_dict!(state_dict) }
      net.eval
      new(net: net, config: config, path: path, state_dict: state_dict)
    rescue LoadError => e
      raise ModelLoadError, "torch-rb not installed (#{e.message}). See README install instructions."
    end

    def self.build_network(config)
      case config.raw["export_kind"]
      when "proxy_cnn"
        ProxyCNN.new(num_classes: config.num_classes)
      when "nnunet_plain_conv_unet"
        BonesUNet.from_config(load_arch_yaml(config))
      else
        raise ModelLoadError,
              "unknown export_kind=#{config.raw['export_kind'].inspect}; " \
              "expected 'nnunet_plain_conv_unet' (real model) or 'proxy_cnn' (legacy)."
      end
    end

    def self.load_arch_yaml(config)
      arch_path = config.arch_yaml_path
      raise ModelLoadError, "missing #{arch_path} (run script/export_totalsegmentator.py --inspect)" \
        unless File.exist?(arch_path)

      require "yaml"
      YAML.safe_load_file(arch_path)
    end

    def initialize(net:, config:, path:, state_dict:)
      @net        = net
      @config     = config
      @path       = path
      @state_dict = state_dict
    end

    # Run the inner CNN on a single patch.
    #
    # @param patch [Torch::Tensor, Numo::SFloat] [1,1,D,H,W] tensor OR [D,H,W] Numo.
    # @return [Torch::Tensor] logits [1, K, D, H, W]
    def forward(patch)
      require "torch"
      tensor =
        case patch
        when Torch::Tensor
          patch.dim == 5 ? patch : patch.unsqueeze(0).unsqueeze(0)
        else
          Torch.from_numo(patch).unsqueeze(0).unsqueeze(0)
        end
      Torch.no_grad { net.forward(tensor) }
    end
  end
end
