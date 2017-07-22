class SimpleConvolutionLayer < LayerBase
  attr_reader :filter
  def initialize w:, h:, size:
    scale = 1.0 / size**2
    @w = w
    @h = h
    @size = size
    @filter = Numo::SFloat.new(size, size).rand(-scale, scale)
  end

  def forward input
    out_w = @w - @size + 1
    out_h = @h - @size + 1
    out_size = out_w * out_h
    temp_size = out_size + @size - 1
    temp_size = @w * @h - (@size - 1) * (@w + 1)
    temp = Numo::SFloat.new(temp_size).fill(0)
    @size.times do |i|
      @size.times do |j|
        f = @filter[i, j]
        offset = j * @w + i
        temp += f * input[offset...(offset + temp_size)]
        p f, input[offset...(offset + temp_size)]
      end
    end
    out = Numo::SFloat.new(out_size).fill(0)
    out_h.times do |j|
      idx = out_w * j
      tidx = @w * j
      out[idx...(idx + out_w)] = temp[tidx...(tidx + out_w)]
    end
    out
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
