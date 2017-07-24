class ChannelSumLayer < LayerBase
  def initialize size
    @size = size
  end

  def forward input_channels
    input_channels.sum
  end

  def backward channels, propagation
    [0, propagation]
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
