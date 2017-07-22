class LayerBase
  attr_accessor :parameter
  def forward input
    raise :unimplemented
  end

  def backward input, propagation
    raise :unimplemented
  end

  def update parameter, grad, delta
    @parameter = parameter - grad * delta if parameter
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
    GradientSet.new(@gradients.zip(set.gradients).map { |a, b| a + b })
  end

  def self.[] *gradients
    new gradients
  end
end
