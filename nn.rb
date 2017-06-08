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

  def backward input, propagation
    [
      Numo::SFloat[propagation.to_a].transpose.dot(Numo::SFloat[input.to_a]),
      @network.transpose.dot(propagation)
    ]
  end
end

class Loss2Layer
  def initialize answer
    @answer = answer
  end

  def forward input
    ((input-@answer)**2).sum
  end

  def backward input, propagation
    [0, 2 * (input - @answer) * propagation]
  end
end

class ActivateLayer
  def forward input
    input.map { |x| activate x }
  end

  def backward input, propagation
    [0, input.map { |x| dactivate x } * propagation]
  end

  def activate x
    Math.log(1+Math.exp(x))
  end

  def dactivate x
    1-1/(1+Math.exp(x))
  end
end

class NN
  def initialize *layers
    @layers = layers
  end

  def forward input
    @inputs = @layers.map do |layer|
      layer_input = input
      input = layer.forward input
      layer_input
    end
    input
  end

  def backward
    propagation = 1
    @layers.zip(@inputs).reverse_each.map do |layer, input|
      _delta, propagation = layer.backward input, propagation
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

input = Numo::SFloat.new(6).seq

nn = NN.new(
  LinearLayer.new(6, 4),
  ActivateLayer.new,
  LinearLayer.new(4, 3),
  ActivateLayer.new,
  Loss2Layer.new(Numo::SFloat.new(3).seq)
)

d = 0.001
igrad = input.size.times.map{|i|
  v=input[i]
  input[i]=v+d
  vp = nn.forward input
  input[i]=v-d
  vm = nn.forward input
  (vp-vm)/2/d
}

lgrad = nn.instance_eval{@layers}.map do |layer|
  next nil unless LinearLayer === layer
  w,h = layer.network.shape
  w.times.map{|i|
    h.times.map{|j|
      v=layer.network[i,j]
      layer.network[i,j]=v+d
      vp = nn.forward input
      layer.network[i,j]=v-d
      vm = nn.forward input
      layer.network[i,j]=v
      (vp-vm)/2/d
    }
  }
end


binding.pry
