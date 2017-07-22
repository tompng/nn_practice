class LinearLayer < LayerBase
  def initialize insize, outsize
    self.parameter = Numo::SFloat.new(outsize, insize+1).rand(-4.0/insize, 4.0/insize)
  end

  def network
    parameter
  end

  def forward input
    network.dot Numo::SFloat[*input,1]
  end

  def backward input, propagation
    input = Numo::SFloat[*input,1]
    prop = propagation.dot network
    [
      Numo::SFloat[propagation.to_a].transpose.dot(Numo::SFloat[input.to_a]),
      Numo::SFloat[*prop.to_a.take(input.size-1)]
    ]
  end
end
