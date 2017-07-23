class CrossEntropyLossLayer < LayerBase
  attr_accessor :answer
  def initialize answer
    @answer = answer
  end

  def forward input
    -input.to_a.zip(@answer).map { |v, t| t.zero? ? 0 : Math.log(v)}.sum
  end

  def backward input, propagation
    [0, Numo::SFloat.asarray(input.to_a.zip(@answer).map { |v, t| t / (1e-10 + v) } * propagation)]
  end
end

class Loss2Layer < LayerBase
  attr_accessor :answer
  def initialize answer
    @answer = answer
  end

  def forward input
    ((input - answer)**2).sum
  end

  def backward input, propagation
    [0, 2 * (input - answer) * propagation]
  end
end
