class LayerBase
  attr_accessor :parameter
  def forward input
    raise :unimplemented
  end

  def forward_with_input_cache input
    [forward(input), input]
  end

  def backward input, propagation
    raise :unimplemented
  end

  def update parameter, grad, delta
    @parameter = parameter - grad * delta if parameter
  end
end

class IdentLayer < LayerBase
  def forward input
    input
  end

  def backward input, propagation
    [nil, propagation]
  end
end

class CompositeLayer < LayerBase
  attr_reader :layers
  def initialize *layers
    @layers = layers.flatten
  end

  def forward_with_input_cache input
    input_cache = @layers.map do |layer|
      input, cache = layer.forward_with_input_cache input
      cache
    end
    [input, input_cache]
  end

  def backward input, propagation
    gradients = @layers.zip(input).reverse_each.map do |layer, input|
      gradient, propagation = layer.backward input, propagation
      gradient
    end
    [GradientSet.new(gradients.reverse), propagation]
  end

  def parameter
    @layers.map &:parameter
  end

  def update parameter, grad, delta
    @layers.zip(parameter, grad.to_ary).each do |layer, p, g|
      layer.update p, g, delta
    end
  end

  def forward input
    @layers.each do |layer|
      input = layer.forward input
    end
  end
end

class GradientSet
  attr_reader :gradients
  def initialize gradients
    @gradients = gradients
  end

  def to_ary
    @gradients
  end

  def + set
    GradientSet.new(@gradients.zip(set.gradients).map { |a, b| a + b if a || b })
  end

  def self.[] *gradients
    new gradients
  end
end
