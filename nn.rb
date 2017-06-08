require 'numo/narray'
require 'pry'

class LinearLayer
  attr_reader :network
  def initialize insize, outsize
    @network = Numo::SFloat.new(outsize, insize).rand(-1, 1)
  end

  def forward input
    @network.dot input
  end

  def update propagation, diff

  end

  def backward input, output
    [@network, input]
  end
end

class Loss2Layer
  def initialize answer
    @answer = answer
  end

  def forward input
    ((input-@answer)**2).sum
  end

  def backward input, output
    [2*(input - @answer), 0]
  end
end

class ActivateLayer
  def forward input
    input.map { |x| activate x }
  end

  def backward input, output
    [input.map { |x| dactivate x }, 0]
  end

  def activate x
    Math.log(1+Math.exp(x))
  end

  def dactivate x
    1-1/(1+Math.exp(x))
  end
end

class NN
  def initialize layers
    @layers = layers
  end

  def forward input
    @layers.inject input do |input, layer|
      layer.forward input
    end
  end

  def batch_learn
    dataset = TrainingDataset.new
    yield dataset

    loss = dataset.map { |input, output|
      err = forward(input) - output
      (err**2).sum
    }.sum
  end

  class TrainingDataset
    attr_reader :dataset

    def initialize
      @dataset = []
    end

    def learn input, output
      @dataset << [input, output]
    end
  end
end

input = Numo::SFloat.new(4).seq
l = LinearLayer.new 4, 3
a = ActivateLayer.new
l2 = Loss2Layer.new Numo::SFloat.new(3).seq

lo = l.forward(input)
ao = a.forward(lo)
l2o = l2.forward(ao)

l2inv = l2.backward(ao, l2o)
ainv = a.backward(lo, ao)
linv = l.backward(input, lo)



d=0.000001
input = Numo::DFloat.new(4).seq
lgrad = Numo::SFloat.new(3,4).fill 0
3.times{|i|4.times{|j|
  v=l.network[i,j]
  l.network[i,j]=v+d
  vp=l2.forward(a.forward(l.forward(input)))
  l.network[i,j]=v-d
  vm=l2.forward(a.forward(l.forward(input)))
  l.network[i,j]=v
  lgrad[i,j]=(vp-vm)/2/d
}}


hoge=Numo::SFloat.new(12,3).fill 0
3.times{|i|
  12.times{|ij|
    hoge[ij,i] = input[ij%4]
  }
}

binding.pry

# d[i, [i,j]] = input[j] -> output[i]
