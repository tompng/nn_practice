class FullConnectedLayer < LayerBase
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

class FullConnectedBiasedLayer < CompositeLayer
  def initialize insize, outsize
    super FullConnectedLayer.new(insize, outsize), BiasLayer.new(outsize)
  end

  def network
    @layers.first.network
  end
end

class SliceLayer < LayerBase
  def initalize offset, length
    @offset = offset
    @length = length
  end

  def forward input
    input[@offset...(@offset + @length)]
  end

  def backward input, propagation
    out = Numo::SFloat.new(input.size).fill(0)
    out[@offset...(@offset + @length)] = propagation
    [nil, out]
  end
end
