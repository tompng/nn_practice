class BiasLayer < LayerBase
  def initialize size, scale: 1 / 16.0
    self.parameter = Numo::SFloat.new(size).rand(-scale, scale)
  end

  def forward input
    input + parameter
  end

  def backward _input, propagation
    [propagation, propagation]
  end
end

class ConstBiasLayer < LayerBase
  def initialize scale: 1 / 16.0
    self.parameter = rand(-scale..scale)
  end

  def forward input
    input + parameter
  end

  def backward _input, propagation
    [propagation.sum, propagation]
  end
end
