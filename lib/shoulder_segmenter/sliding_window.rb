# frozen_string_literal: true

require "numo/narray"

module ShoulderSegmenter
  # nnU-Net-style 3-D sliding-window inference with Gaussian importance blending.
  #
  # 1. Reflect-pad the volume so each spatial dim >= patch_size.
  # 2. Step through with stride = floor(patch_size * (1 - overlap)) (default 0.5).
  # 3. For each patch, run model.forward → logits [1,K,D,H,W].
  # 4. Multiply by a 3-D Gaussian importance map (sigma = patch_size / 8) and
  #    accumulate into a logit volume + weight volume.
  # 5. logits / weights, argmax over K, crop pad → final label volume.
  module SlidingWindow
    module_function

    OVERLAP = 0.5
    GAUSSIAN_SIGMA_SCALE = 1.0 / 8

    def gaussian_importance_map(patch_size)
      sigmas = patch_size.map { |s| s * GAUSSIAN_SIGMA_SCALE }
      gz = gaussian_1d(patch_size[0], sigmas[0])
      gy = gaussian_1d(patch_size[1], sigmas[1])
      gx = gaussian_1d(patch_size[2], sigmas[2])
      g = gz.reshape(patch_size[0], 1, 1) *
          gy.reshape(1, patch_size[1], 1) *
          gx.reshape(1, 1, patch_size[2])
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
      orig_shape = volume.shape

      padded, _pads = reflect_pad_to(volume, patch_size)
      pd, ph, pw = padded.shape
      sz, sy, sx = strides(patch_size)

      k       = model.config.num_classes
      logits  = Numo::SFloat.zeros(k, pd, ph, pw)
      weights = Numo::SFloat.zeros(pd, ph, pw)
      gmap    = gaussian_importance_map(patch_size)

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
            out   = model.forward(patch)              # [1,K,D,H,W]
            out_arr = out.squeeze(0).numo             # [K,D,H,W] Numo::SFloat

            k.times do |c|
              logits[c, z0...z1, y0...y1, x0...x1] += out_arr[c, true, true, true] * gmap
            end
            weights[z0...z1, y0...y1, x0...x1] += gmap
          end
        end
      end

      labels_padded = argmax_channel(logits)
      crop(labels_padded, orig_shape).cast_to(Numo::UInt8)
    end

    # Argmax along the channel (axis 0) of a [K, D, H, W] SFloat. Numo's
    # `max_index(axis: 0)` returns **flat** indices into the source array, not
    # per-axis indices, so divide by the per-axis-0 stride (D*H*W) to recover
    # the channel id.
    def argmax_channel(logits)
      _k, d, h, w = logits.shape
      flat = logits.max_index(axis: 0)
      flat / (d * h * w)
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

    # Reflect-pad volume up to at least patch_size in every dim. This mirrors
    # nnU-Net's preprocessing (and avoids the dark zero-band that biases the
    # CT-normalized voxel statistics inside InstanceNorm at the boundary).
    def reflect_pad_to(arr, patch_size)
      d, h, w = arr.shape
      pad_d = [patch_size[0] - d, 0].max
      pad_h = [patch_size[1] - h, 0].max
      pad_w = [patch_size[2] - w, 0].max
      return [arr, [0, 0, 0]] if pad_d.zero? && pad_h.zero? && pad_w.zero?

      [reflect_pad_3d(arr, pad_d, pad_h, pad_w), [pad_d, pad_h, pad_w]]
    end

    # Edge-of-axis reflect: post-pad only (we never crop from the front, so we
    # only need to reflect the tail). For axis dim d, indices d..(d+pad-1)
    # reflect back into the volume as d-2, d-3, ….
    def reflect_pad_3d(arr, pad_d, pad_h, pad_w)
      d, h, w = arr.shape
      out = Numo::SFloat.zeros(d + pad_d, h + pad_h, w + pad_w)
      out[0...d, 0...h, 0...w] = arr

      # Reflect along d
      pad_d.times do |i|
        src = (d - 2 - i) % d
        out[d + i, 0...h, 0...w] = arr[src, true, true]
      end if pad_d.positive?

      # Reflect along h (across the already-d-extended volume)
      if pad_h.positive?
        ext_d = d + pad_d
        pad_h.times do |i|
          src = (h - 2 - i) % h
          out[0...ext_d, h + i, 0...w] = out[0...ext_d, src, 0...w]
        end
      end

      # Reflect along w
      if pad_w.positive?
        ext_d = d + pad_d
        ext_h = h + pad_h
        pad_w.times do |i|
          src = (w - 2 - i) % w
          out[0...ext_d, 0...ext_h, w + i] = out[0...ext_d, 0...ext_h, src]
        end
      end

      out
    end

    def crop(arr, original_shape)
      d, h, w = original_shape
      arr[0...d, 0...h, 0...w]
    end
  end
end
