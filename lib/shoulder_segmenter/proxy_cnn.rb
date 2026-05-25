# frozen_string_literal: true

module ShoulderSegmenter
  # Ruby mirror of the proxy CNN defined in `script/export_totalsegmentator.py`.
  #
  #   Conv3d(in=1,  out=4,           k=3, pad=1, bias=true)
  #   ReLU
  #   Conv3d(in=4,  out=num_classes, k=3, pad=1, bias=true)
  #
  # Trivial on purpose: Phase 2 only needs to validate the round-trip
  # (Python state_dict → Ruby state_dict → forward pass matches bit-for-bit).
  # The real nnU-Net mirror lands in Phase 3 with the same load-weights flow.
  class ProxyCNN < (begin
    require "torch"
    Torch::NN::Module
  end)
    def initialize(num_classes:)
      super()
      @conv1 = Torch::NN::Conv3d.new(1, 4,           3, padding: 1, bias: true)
      @relu  = Torch::NN::ReLU.new
      @conv2 = Torch::NN::Conv3d.new(4, num_classes, 3, padding: 1, bias: true)
    end

    def forward(x)
      @conv2.call(@relu.call(@conv1.call(x)))
    end

    # Copy weights from a state_dict (Hash<String, Torch::Tensor>) into the
    # parallel parameter slots. Keys mirror the Python attribute names.
    def load_state_dict!(sd)
      Torch.no_grad do
        @conv1.weight.copy!(sd.fetch("conv1.weight"))
        @conv1.bias.copy!(sd.fetch("conv1.bias"))
        @conv2.weight.copy!(sd.fetch("conv2.weight"))
        @conv2.bias.copy!(sd.fetch("conv2.bias"))
      end
      self
    end
  end
end
