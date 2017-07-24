class SimpleConvolutionLayer < LayerBase
  attr_reader :filter
  def initialize w:, h:, size:
    scale = 1.0 / size**2
    @w = w
    @h = h
    @size = size
    self.parameter = Numo::SFloat.new(size, size).rand(-scale, scale)
  end

  def filter
    parameter
  end

  def forward input
    out_w = @w - @size + 1
    out_h = @h - @size + 1
    out_size = out_w * out_h
    temp_size = @w * @h - (@size - 1) * (@w + 1)
    temp = Numo::SFloat.new(temp_size).fill(0)
    (0...@size).to_a.repeated_permutation(2) do |i, j|
      f = filter[i, j]
      offset = j * @w + i
      temp += f * input[offset...(offset + temp_size)]
    end
    out = Numo::SFloat.new(out_size).fill(0)
    out_h.times do |j|
      idx = out_w * j
      tidx = @w * j
      out[idx...(idx + out_w)] = temp[tidx...(tidx + out_w)]
    end
    out
  end

  def backward input, propagation
    pw = @w - @size + 1
    ph = @h - @size + 1
    temp = Numo::SFloat.new(@w * @h).fill(0)
    ph.times do |y|
      temp[(@w * y)...(@w * y + pw)] = propagation[(pw * y)...(pw * y + pw)]
    end
    out = Numo::SFloat.new(@w * @h).fill(0)
    (0...@size).to_a.repeated_permutation(2) do |i, j|
      f = filter[i, j]
      offset = j * @w + i
      length = @w * (ph - 1) + pw
      out[offset...(offset + length)] += f * temp[0...length]
    end
    grad = Numo::SFloat.new(@size, @size).fill(0)
    (0...@size).to_a.repeated_permutation(2) do |i, j|
      ph.times.map do |y|
        offset = @w * (y + j) + i
        grad[i, j] += input[offset...(offset + pw)].sum
      end
    end
    [grad, out]
  end
end

class PaddingLayer < LayerBase
  def initialize w:, h:, padding:, with: 0
    @w = w
    @h = h
    @padding = padding
    @with = with
  end

  def forward input
    w2 = @w + 2 * @padding
    h2 = @h + 2 * @padding
    Numo::SFloat.new(w2 * h2).fill(@with).tap do |out|
      @h.times do |y|
        offset = w2 * (y + @padding) + @padding
        out[offset...(offset + @w)] = input[(@w * y)...(@w * y + @w)]
      end
    end
  end

  def backward _input, propagation
    w2 = @w + 2 * @padding
    out = Numo::SFloat.new(@w * @h).fill(0)
    @h.times do |y|
      offset = w2 * (y + @padding) + @padding
        out[(@w * y)...(@w * y + @w)] = propagation[offset...(offset + @w)]
    end
    [0, out]
  end
end


class MaxPoolingLayer < LayerBase
  def initialize in_w:, in_h:, out_w:, out_h:, pool:
    @in_w = in_w
    @in_h = in_h
    @out_w = out_w
    @out_h = out_h
    @pool = pool
    @stride_w = (in_w - pool) / (out_w - 1)
    @stride_h = (in_h - pool) / (out_h - 1)
  end

  def forward input
    out = Numo::SFloat.new(@out_w * @out_h).fill(0)
    @out_w.times do |i|
      ii = @stride_w * i
      @out_h.times do |j|
        jj = @stride_h * j
        pools = Array.new @pool do |s|
          idx = (jj + s) * @in_w + ii
          input[idx...(idx + @pool)].max
        end
        out[@out_w * j + i] = pools.max
      end
    end
    out
  end
  def backward input, propagation
    out = Numo::SFloat.new(@in_w * @in_h).fill(0)
    indices = @pool.times.flat_map do |i|
      Array.new(@pool) { |j| j * @in_w + i }
    end
    @out_w.times do |i|
      ii = @stride_w * i
      @out_h.times do |j|
        jj = @stride_h * j
        idx = jj * @in_w + ii
        values = indices.map { |idx2| input[idx + idx2] }
        max_index = idx + indices[values.index(values.max)]
        out[max_index] += propagation[@out_w * j + i]
      end
    end
    [0, out]
  end
end
