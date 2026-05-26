# frozen_string_literal: true

require "set"
require "torch"

module ShoulderSegmenter
  # Ruby mirror of TotalSegmentator's `total --fast` (task 297) nnU-Net:
  # `dynamic_network_architectures.architectures.unet.PlainConvUNet`.
  #
  # Architecture (sourced from script/golden/nnunet_architecture.yaml; pulled
  # straight out of TotalSegmentator's nnUNetPlans):
  #   - 5 encoder stages, features [32, 64, 128, 256, 320]
  #   - 4 decoder stages (encoder_stages - 1)
  #   - Each stage holds 2 ConvNormReLU blocks (Conv3d → InstanceNorm3d → LeakyReLU(0.01))
  #   - Encoder downsample: strided Conv3d on the first conv of each stage
  #     (stride [1,1,1] for stage 0, [2,2,2] for stages 1-4)
  #   - Decoder upsample: ConvTranspose3d with kernel=2 stride=2
  #   - Skip connections: concat with the matching encoder stage output
  #   - Output: 1x1x1 Conv3d (32 → output_channels) at the top resolution.
  #     Deep supervision is OFF for inference, so only seg_layers[-1] fires.
  #
  # State-dict keys (88, matches the Python checkpoint after stripping the
  # `all_modules.*` aliases and shared `decoder.encoder.*` references):
  #
  #   encoder.stages.<S>.0.convs.<C>.conv.{weight,bias}    S in 0..4, C in 0..1
  #   encoder.stages.<S>.0.convs.<C>.norm.{weight,bias}
  #   decoder.stages.<S>.convs.<C>.conv.{weight,bias}      S in 0..3, C in 0..1
  #   decoder.stages.<S>.convs.<C>.norm.{weight,bias}
  #   decoder.transpconvs.<S>.{weight,bias}                S in 0..3
  #   decoder.seg_layers.<S>.{weight,bias}                 S in 0..3
  class BonesUNet < Torch::NN::Module
    # ---------------------------------------------------------------------
    # Building blocks
    # ---------------------------------------------------------------------

    # Conv3d → InstanceNorm3d → LeakyReLU. Mirrors nnU-Net's
    # `ConvDropoutNormReLU`. Attribute names are `conv` and `norm` so the
    # state_dict keys are `…convs.<C>.conv.weight` etc.
    class ConvNormReLU < Torch::NN::Module
      attr_reader :conv, :norm

      def initialize(in_ch:, out_ch:, kernel:, stride:, padding:,
                     norm_eps: 1e-5, leaky_slope: 0.01)
        super()
        @conv = Torch::NN::Conv3d.new(in_ch, out_ch, kernel,
                                      stride: stride, padding: padding, bias: true)
        @norm = Torch::NN::InstanceNorm3d.new(out_ch, eps: norm_eps, affine: true)
        @leaky_slope = leaky_slope
      end

      def forward(x)
        Torch::NN::Functional.leaky_relu(@norm.call(@conv.call(x)), @leaky_slope, inplace: false)
      end
    end

    # `StackedConvBlocks` in nnU-Net. Holds a Sequential of ConvNormReLU blocks
    # at attribute `convs`, so state_dict keys land as `…convs.<C>.…`.
    class StackedConvs < Torch::NN::Module
      attr_reader :convs

      def initialize(blocks:)
        super()
        @convs = Torch::NN::Sequential.new(*blocks)
      end

      def forward(x)
        @convs.call(x)
      end
    end

    # 3D transposed convolution. torch-rb 0.24.0 doesn't expose ConvTranspose3d
    # as a Module (only Conv1d/2d/3d and ConvTranspose 1d/2d are wrapped), but
    # the underlying `Torch.conv_transpose3d` op is bound, so we wrap it. Param
    # names match PyTorch's ConvTranspose3d state_dict — `weight` shape
    # [in_ch, out_ch/groups, kT, kH, kW], `bias` shape [out_ch].
    class ConvTranspose3d < Torch::NN::Module
      attr_reader :weight, :bias

      def initialize(in_ch:, out_ch:, kernel:, stride:)
        super()
        kt, kh, kw = Array(kernel).then { |a| a.length == 3 ? a : [a, a, a] }
        @stride         = Array(stride).then { |a| a.length == 3 ? a : [a, a, a] }
        @padding        = [0, 0, 0]
        @output_padding = [0, 0, 0]
        @groups         = 1
        @dilation       = [1, 1, 1]
        @weight = Torch::NN::Parameter.new(Torch.empty(in_ch, out_ch, kt, kh, kw))
        @bias   = Torch::NN::Parameter.new(Torch.empty(out_ch))
        Torch::NN::Init.kaiming_uniform!(@weight, a: Math.sqrt(5))
        Torch::NN::Init.zeros!(@bias)
      end

      def forward(x)
        Torch.conv_transpose3d(x, @weight, @bias, @stride, @padding,
                               @output_padding, @groups, @dilation)
      end
    end

    # ---------------------------------------------------------------------
    # Encoder / Decoder shells
    # ---------------------------------------------------------------------

    class Encoder < Torch::NN::Module
      attr_reader :stages

      def initialize(stages_list)
        super()
        # nnU-Net wraps each StackedConvBlocks in a Sequential of length 1
        # (an artifact of how DownsampleBlock used to live there). We mirror
        # that wrapping exactly so the state_dict path `…stages.<S>.0.convs.<C>.…`
        # resolves with the leading `0` index.
        wrapped = stages_list.map { |stacked| Torch::NN::Sequential.new(stacked) }
        @stages = Torch::NN::Sequential.new(*wrapped)
      end

      def forward(x)
        skips = []
        # iterate via the underlying Hash since Sequential exposes `each` but
        # in a way that drops indices.
        modules_hash = @stages.instance_variable_get(:@modules)
        modules_hash.keys.sort_by(&:to_i).each do |k|
          x = modules_hash[k].call(x)
          skips << x
        end
        skips
      end
    end

    class Decoder < Torch::NN::Module
      attr_reader :stages, :transpconvs, :seg_layers

      def initialize(stages:, transpconvs:, seg_layers:)
        super()
        @stages      = Torch::NN::ModuleList.new(stages)
        @transpconvs = Torch::NN::ModuleList.new(transpconvs)
        @seg_layers  = Torch::NN::ModuleList.new(seg_layers)
      end

      # nnU-Net UNetDecoder.forward with deep_supervision=False — only the
      # top-resolution seg head emits a tensor.
      def forward(skips)
        lres = skips[-1]
        n = @stages.length
        n.times do |s|
          up   = @transpconvs[s].call(lres)
          skip = skips[-(s + 2)]
          x = Torch.cat([up, skip], 1)
          lres = @stages[s].call(x)
        end
        @seg_layers[n - 1].call(lres)
      end
    end

    # ---------------------------------------------------------------------
    # PlainConvUNet itself
    # ---------------------------------------------------------------------

    def self.from_config(arch_config)
      new(
        n_stages:                arch_config.fetch("n_stages"),
        features_per_stage:      arch_config.fetch("features_per_stage"),
        strides:                 arch_config.fetch("strides"),
        kernel_sizes:            arch_config.fetch("kernel_sizes"),
        n_conv_per_stage:        arch_config.fetch("n_conv_per_stage"),
        n_conv_per_stage_decoder: arch_config.fetch("n_conv_per_stage_decoder"),
        input_channels:          arch_config.fetch("input_channels", 1),
        output_channels:         arch_config.fetch("output_channels"),
        norm_eps:                arch_config.dig("norm_op_kwargs", "eps") || 1e-5,
        leaky_slope:             arch_config.dig("nonlin_kwargs", "negative_slope") || 0.01
      )
    end

    attr_reader :encoder, :decoder

    def initialize(n_stages:, features_per_stage:, strides:, kernel_sizes:,
                   n_conv_per_stage:, n_conv_per_stage_decoder:,
                   output_channels:, input_channels: 1,
                   norm_eps: 1e-5, leaky_slope: 0.01)
      super()
      raise ArgumentError, "encoder stage counts mismatch" \
        unless [features_per_stage.length, strides.length, kernel_sizes.length,
                n_conv_per_stage.length].all? { |n| n == n_stages }
      raise ArgumentError, "decoder needs n_stages - 1 stages" \
        unless n_conv_per_stage_decoder.length == n_stages - 1

      # ---- encoder stages -------------------------------------------------
      encoder_stages = []
      in_ch = input_channels
      n_stages.times do |s|
        out_ch  = features_per_stage[s]
        stride  = strides[s]
        kernel  = kernel_sizes[s]
        padding = kernel.map { |k| k / 2 }
        unit_stride = stride.map { 1 }
        blocks = []
        n_conv_per_stage[s].times do |c|
          blocks << ConvNormReLU.new(
            in_ch:       c == 0 ? in_ch : out_ch,
            out_ch:      out_ch,
            kernel:      kernel,
            stride:      c == 0 ? stride : unit_stride,
            padding:     padding,
            norm_eps:    norm_eps,
            leaky_slope: leaky_slope
          )
        end
        encoder_stages << StackedConvs.new(blocks: blocks)
        in_ch = out_ch
      end
      @encoder = Encoder.new(encoder_stages)

      # ---- decoder stages + transpconvs + seg_layers ----------------------
      decoder_stages = []
      transpconvs    = []
      seg_layers     = []
      (n_stages - 1).times do |s|
        # decoder stage s feeds from the (n_stages - 1 - s)-th encoder stage and
        # concats with the higher-resolution skip from encoder stage (n_stages - 2 - s).
        skip_idx = n_stages - 2 - s
        deep_ch  = features_per_stage[n_stages - 1 - s]
        skip_ch  = features_per_stage[skip_idx]

        # The transpconv reverses the stride that *entered* the next-deeper
        # encoder stage. For nnU-Net the upsample kernel equals the stride.
        up_stride = strides[skip_idx + 1]
        transpconvs << ConvTranspose3d.new(
          in_ch:  deep_ch,
          out_ch: skip_ch,
          kernel: up_stride,
          stride: up_stride
        )

        # After concat the channel count doubles (upsampled + skip).
        in_ch = skip_ch * 2
        kernel  = kernel_sizes[skip_idx]
        padding = kernel.map { |k| k / 2 }
        unit_stride = kernel.map { 1 }
        blocks = []
        n_conv_per_stage_decoder[s].times do |c|
          blocks << ConvNormReLU.new(
            in_ch:       c == 0 ? in_ch : skip_ch,
            out_ch:      skip_ch,
            kernel:      kernel,
            stride:      unit_stride,
            padding:     padding,
            norm_eps:    norm_eps,
            leaky_slope: leaky_slope
          )
        end
        decoder_stages << StackedConvs.new(blocks: blocks)
        seg_layers << Torch::NN::Conv3d.new(skip_ch, output_channels, 1, bias: true)
      end

      @decoder = Decoder.new(stages: decoder_stages,
                             transpconvs: transpconvs,
                             seg_layers: seg_layers)
    end

    def forward(x)
      skips = @encoder.call(x)
      @decoder.call(skips)
    end

    # ---- state_dict loader ----------------------------------------------
    #
    # The pickled state_dict from `script/export_totalsegmentator.py` has 88
    # canonical keys (the `all_modules.*` aliases and `decoder.encoder.*`
    # shared-reference duplicates have already been stripped Python-side).
    def load_state_dict!(sd)
      Torch.no_grad do
        sd.each do |key, tensor|
          slot = resolve_param(key)
          raise ModelLoadError, "no parameter slot for state_dict key #{key.inspect}" if slot.nil?
          unless slot.size == tensor.size
            raise ModelLoadError,
                  "shape mismatch for #{key}: " \
                  "module=#{slot.size.inspect} state_dict=#{tensor.size.inspect}"
          end
          slot.copy!(tensor)
        end
      end
      verify_all_params_loaded!(sd.keys)
      self
    end

    private

    # Walk dotted keys into the live Parameter tensor. Numeric path segments
    # index into Sequential/ModuleList children (via the underlying @modules Hash).
    def resolve_param(key)
      node  = self
      parts = key.split(".")
      until parts.empty?
        seg = parts.shift
        node = next_node(node, seg, last: parts.empty?)
        return nil if node.nil?
      end
      node.is_a?(Torch::Tensor) ? node : nil
    end

    def next_node(node, seg, last:)
      if seg =~ /\A\d+\z/
        mods = node.instance_variable_get(:@modules)
        return nil unless mods.is_a?(Hash)
        return mods[seg]
      end

      if last
        # Parameter leaves live as @weight / @bias instance vars on torch-rb
        # modules; reading via the accessor returns the actual tensor.
        return node.weight if seg == "weight" && node.respond_to?(:weight)
        return node.bias   if seg == "bias"   && node.respond_to?(:bias)
      end

      # Submodule by attribute. Both our own modules (Conv3d wrappers, etc.) and
      # torch-rb's Conv3d store submodules as `@<name>` and also surface them
      # through the `@modules` hash, so check both.
      ivar = "@#{seg}"
      if node.instance_variable_defined?(ivar)
        v = node.instance_variable_get(ivar)
        return v unless v.is_a?(Hash)
      end
      mods = node.instance_variable_get(:@modules)
      mods[seg] if mods.is_a?(Hash)
    end

    def verify_all_params_loaded!(loaded_keys)
      seen = Set.new(loaded_keys)
      missing = named_parameters.reject { |name, _| seen.include?(name) }.map(&:first)
      return if missing.empty?

      raise ModelLoadError,
            "state_dict missing #{missing.length} parameter(s): " \
            "#{missing.first(5).inspect}#{'…' if missing.length > 5}"
    end
  end
end
