class LinearLayer < LayerBase
  def initialize insize, outsize, scale: 1 / insize**0.5
    self.parameter = Numo::SFloat.new(outsize, insize).rand(-scale, scale)
  end

  def network
    parameter
  end

  def forward input
    network.dot input
  end

  def backward input, propagation
    [
      Numo::SFloat[propagation.to_a].transpose.dot(Numo::SFloat[input.to_a]),
      propagation.dot(network)
    ]
  end
end

class LinearWithBiasLayer < LayerBase
  def initialize insize, outsize
    @linear_layer = LinearLayer.new insize, outsize
    @bias_layer = BiasLayer.new outsize
  end

  def network
    @linear_layer.network
  end

  def parameter
    [@linear_layer.parameter, @bias_layer.parameter]
  end

  def update parameter, grad, delta
    lp, bp = parameter
    lg, bg = grad
    @linear_layer.update lp, lg, delta
    @bias_layer.update bp, bg, delta if $aaa
  end

  def forward input
    @bias_layer.forward @linear_layer.forward(input)
  end

  def backward input, propagation
    bgrad, bprop = @bias_layer.backward nil, propagation
    lgrad, lprop = @linear_layer.backward input, bprop
    [GradientSet[lgrad, bgrad], lprop]
  end
end

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
