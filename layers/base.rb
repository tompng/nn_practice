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
