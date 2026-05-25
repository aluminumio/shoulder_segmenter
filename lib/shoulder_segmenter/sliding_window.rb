# frozen_string_literal: true

require "numo/narray"

module ShoulderSegmenter
  # nnU-Net-style 3-D sliding-window inference with Gaussian importance blending.
  #
  # 1. Pad the volume so each spatial dim >= patch_size (reflect padding).
  # 2. Step through with stride = floor(patch_size * (1 - overlap)) (default 0.5).
  # 3. For each patch, run model.forward → logits [1,K,D,H,W].
  # 4. Multiply by a 3-D Gaussian importance map (sigma = patch_size / 8) and
  #    accumulate into a logit volume + weight volume.
  # 5. logits / weights, argmax over K, crop pad → final label volume.
  #
  # Phase 2: returns the same shape as input; resampling-back-to-input-spacing
  # is the caller's job (or another TODO).
  module SlidingWindow
    module_function

    OVERLAP = 0.5
    GAUSSIAN_SIGMA_SCALE = 1.0 / 8

    def gaussian_importance_map(patch_size)
      sigmas = patch_size.map { |s| s * GAUSSIAN_SIGMA_SCALE }
      # Separable Gaussian: outer-product of three 1-D Gaussians.
      gz = gaussian_1d(patch_size[0], sigmas[0])
      gy = gaussian_1d(patch_size[1], sigmas[1])
      gx = gaussian_1d(patch_size[2], sigmas[2])
      g = gz.reshape(patch_size[0], 1, 1) * gy.reshape(1, patch_size[1], 1) * gx.reshape(1, 1, patch_size[2])
      g / g.max
    end

    def gaussian_1d(n, sigma)
      center = (n - 1) / 2.0
      xs = Numo::SFloat.new(n).seq
      Numo::NMath.exp(-((xs - center)**2) / (2.0 * sigma * sigma))
    end

    # @param volume [Numo::SFloat] [D,H,W] preprocessed volume
    # @param model  [ShoulderSegmenter::Model]
    # @return [Numo::UInt8] [D,H,W] argmax labels
    def run(volume, model:)
      require "torch"
      patch_size = Array(model.config.patch_size)
      shape      = volume.shape

      # Pad each dim up to at least patch_size (mirror pad) — keep originals for crop.
      padded, pads = pad_to(volume, patch_size)

      pd, ph, pw = padded.shape
      sz, sy, sx = strides(patch_size)

      k         = model.config.num_classes
      logits    = Numo::SFloat.zeros(k, pd, ph, pw)
      weights   = Numo::SFloat.zeros(pd, ph, pw)
      gmap      = gaussian_importance_map(patch_size)

      starts_d = patch_starts(pd, patch_size[0], sz)
      starts_h = patch_starts(ph, patch_size[1], sy)
      starts_w = patch_starts(pw, patch_size[2], sx)

      starts_d.each do |z0|
        starts_h.each do |y0|
          starts_w.each do |x0|
            z1 = z0 + patch_size[0]
            y1 = y0 + patch_size[1]
            x1 = x0 + patch_size[2]

            patch = padded[z0...z1, y0...y1, x0...x1]
            out   = model.forward(patch) # Torch [1,K,D,H,W]
            out_arr = out.squeeze(0).numo # [K,D,H,W] Numo::SFloat

            k.times do |c|
              logits[c, z0...z1, y0...y1, x0...x1] += out_arr[c, true, true, true] * gmap
            end
            weights[z0...z1, y0...y1, x0...x1] += gmap
          end
        end
      end

      # Normalize, argmax, crop pad.
      weights_safe = weights + 1e-8
      # broadcast normalization: logits[:, z, y, x] / weights[z,y,x]
      k.times { |c| logits[c, true, true, true] = logits[c, true, true, true] / weights_safe }
      labels_padded = logits.max_index(axis: 0) % k # argmax along channel
      # max_index returns flat indices; we used axis: 0 which is the channel axis.
      # For NArray, max_index(axis: 0) returns indices within the reduced axis dimension already.

      crop(labels_padded, pads, shape).cast_to(Numo::UInt8)
    end

    def strides(patch_size)
      patch_size.map { |s| [(s * (1 - OVERLAP)).floor, 1].max }
    end

    def patch_starts(extent, patch, stride)
      return [0] if extent <= patch

      starts = (0..(extent - patch)).step(stride).to_a
      starts << (extent - patch) unless starts.last == extent - patch
      starts.uniq
    end

    # Reflect-pad volume up to at least patch_size in every dim.
    def pad_to(arr, patch_size)
      d, h, w = arr.shape
      pd = [patch_size[0] - d, 0].max
      ph = [patch_size[1] - h, 0].max
      pw = [patch_size[2] - w, 0].max
      return [arr, [0, 0, 0]] if pd.zero? && ph.zero? && pw.zero?

      # Simple zero pad (TODO Phase 3: reflect pad matching nnU-Net exactly).
      padded = Numo::SFloat.zeros(d + pd, h + ph, w + pw)
      padded[0...d, 0...h, 0...w] = arr
      [padded, [pd, ph, pw]]
    end

    def crop(arr, _pads, original_shape)
      d, h, w = original_shape
      arr[0...d, 0...h, 0...w]
    end
  end
end
