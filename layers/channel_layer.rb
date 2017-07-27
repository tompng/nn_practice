class ChannelSumLayer < LayerBase
  def forward input_channels
    input_channels.sum
  end

  def backward input_channels, propagation
    [nil, input_channels.map { propagation }]
  end
end

class ChannelMixLayer < LayerBase
  def initialize size
    @size = size
    scale = 1.0 / size
    self.parameter = Numo::SFloat.new(size).fill(-scale, scale)
  end

  def forward input_channels
    input_channels.zip(parameter.to_a).map { |c, p| c * p }.sum
  end

  def backward input_channels, propagation
    grad = Numo::SFloat.new(@size).fill(0)
    @size.times do |i|
      grad[i] = input_channels[i].dot propagation
    end
    propagations = parameter.map { |param| propagation * param }
    [grad, propagations]
  end
end

class ChannelSplitLayer < LayerBase
  def initialize *layers
    @layers = layers.flatten
  end

  def forward input
    @layers.map do |layer|
      layer.forward input
    end
  end

  def backward input, propagation
    grad_propagations = @layers.map do |layer|
      layer.backward input, propagation
    end
    grad = GradientSet.new(grad_propagations.map(&:first))
    prop = grad_propagations.map(&:last).sum
    [grad, prop]
  end
end

class MultiChannelConvolutionLayer < LayerBase
  def initialize w:, h:, size:, insize:, outsize:
    @insize = insize
    @outsize = outsize
    @layer_matrix = Array.new outsize do
      Array.new insize do
        SimpleConvolutionLayer.new w: w, h: h, size: size
      end
    end
    @sum_layer = ChannelSumLayer.new
  end

  def forward input_channels
    @layer_matrix.map do |layers|
      outputs = layers.zip(input_channels).map do |input, layer|
        layer.forward input
      end
      outputs.sum
    end
  end

  def parameter
    @layer_matrix.map { |layers| layers.map &:parameter }
  end

  def update parameters, grads, delta
    @layer_matrix.each_with_index do |layers, i|
      layers.each_with_index do |layer, j|
        param = parameters[i][j]
        grad = grads[j * @layer_matrix.size + i]
        layer.update param, grad, delta
      end
    end
  end

  def backward input_channels, propagation_channels
    grads = []
    props = []
    @layer_matrix.zip(propagation_channels).with_index do |(layers, propagation), i|
      layers.zip(input_channels).with_index do |(layer, input), j|
        grad, prop = layer.backward input, propagation
        props[j] = props[j] ? props[j] + prop : prop
        grads[j * @layer_matrix.size + i] = grad
      end
    end
    [GradientSet.new(grads), props]
  end
end

class ChannelMapLayer < LayerBase
  def initialize layers = nil, size: nil, layer: nil
    @layers = layers and return if layers
    raise unless layer.parameter.nil?
    @layers = size.times.map { layer }
  end

  def forward input_channels
    @layers.zip(input_channels).map do |layer, input|
      layer.forward input
    end
  end

  def parameter
    @layers.map &:parameters
  end

  def update parameters, grads, delta
    @layers.zip(parameters, grads).each do |layer, param, grad|
      layer.update param, grad, delta
    end
  end

  def backward input_channels, propagation_channels
    grad_props = @layers.zip(input_channels, propagation_channels).map do |layer, input, propagation|
      layer.backward input, propagation
    end
    [GradientSet.new(grad_props.map &:first), grad_props.map(&:last)]
  end
end

class VectorToSingleChannelLayer < LayerBase
  def forward input
    [input]
  end

  def backward _input, propagation_channels
    [nil, propagation_channels.first]
  end
end

class SingleChannelToVectorLayer < LayerBase
  def forward input_channels
    input_channels.first
  end

  def backward _input, propagation
    [nil, [propagation]]
  end
end

class ChannelConcatLayer < LayerBase
  def forward inputs
    total_size = inputs.map(&:size).sum
    out = Numo::SFloat.new(total_size).fill(0)
    inputs.reduce 0 do |offset, input|
      out[offset...(offset + input.size)] = input
      offset + input.size
    end
    out
  end

  def backward inputs, propagation
    propagations = []
    inputs.reduce 0 do |offset, input|
      propagations << propagation[offset...(offset + input.size)]
      offset + input.size
    end
    [nil, propagations]
  end
end
