# frozen_string_literal: true

require_relative "proxy_cnn"

module ShoulderSegmenter
  # Wrapper around the inner CNN. Phase 2 ships the *proxy* CNN — a tiny
  # 2-layer Conv3d sandwich — defined in both Python (`script/export_totalsegmentator.py`)
  # and Ruby (`proxy_cnn.rb`), with weights round-tripped via libtorch's
  # `torch::pickle_save` / `torch::pickle_load` (Ruby `Torch.load`).
  #
  # Input  shape: [B=1, C=1, D, H, W] float32 (D/H/W = config.patch_size)
  # Output shape: [B=1, K, D, H, W]  float32 logits (K = config.num_classes)
  #
  # Phase 3 substitutes the real nnU-Net mirror without changing this surface.
  class Model
    CACHE_DIR = File.expand_path("~/.cache/shoulder_segmenter")

    attr_reader :net, :config, :path

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
      new(net: net, config: config, path: path, state_dict: state_dict)
    rescue LoadError => e
      raise ModelLoadError, "torch-rb not installed (#{e.message}). See README install instructions."
    end

    def self.build_network(config)
      case config.raw["export_kind"]
      when "proxy_cnn"
        ProxyCNN.new(num_classes: config.num_classes)
      else
        raise ModelLoadError,
              "unknown export_kind=#{config.raw['export_kind'].inspect}. " \
              "Phase 2 only ships 'proxy_cnn'; the real nnU-Net mirror is Phase 3."
      end
    end

    attr_reader :state_dict

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
      net.forward(tensor)
    end
  end
end
