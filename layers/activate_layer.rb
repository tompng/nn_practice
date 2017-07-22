class ActivateLayerBase < LayerBase
  def forward input
    input.map { |x| activate x }
  end

  def backward input, propagation
    [0, input.map { |x| dactivate x } * propagation]
  end
end

class SigmoidLayer < ActivateLayerBase
  def activate x
    v = 1-1/(1+Math.exp(x))
    v.finite? ? v : (x < 0 ? 0 : 1)
  end

  def dactivate x
    v = Math.exp(x)/(1+Math.exp(x))**2
    v.finite? ? v : 0
  end
end

class SoftplusLayer < ActivateLayerBase
  def activate x
    v = Math.log(1+Math.exp(x))
    v.finite? ? v : (x < 0 ? 0 : x)
  end

  def dactivate x
    v = 1-1/(1+Math.exp(x))
    v.finite? ? v : (x < 0 ? 0 : 1)
  end
end

class ReLULayer <ActivateLayerBase
  def activate x
    x < 0 ? 0 : x
  end

  def dactivate x
    x < 0 ? 0 : 1
  end
end

class SoftmaxLayer < LayerBase
  def forward input
    max = input.max
    exps = input.map { |v| Math.exp(v - max) }
    sum = exps.sum
    exps.map { |v| v / sum }
  end

  def backward input, propagation
    max = input.max
    exps = input.map { |v| Math.exp(v - max) }
    sum2 = exps.sum**2
    ep = exps * propagation
    [0, (ep + ep.sum * exps) / sum2]
  end
end
