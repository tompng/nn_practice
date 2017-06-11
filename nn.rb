require 'numo/narray'
require 'pry'

class LinearLayer
  attr_accessor :network
  def initialize insize, outsize
    @network = Numo::SFloat.new(outsize, insize+1).rand(-4, 4)
  end

  def forward input
    @network.dot Numo::SFloat[*input,1]
  end

  def backward input, propagation
    input = Numo::SFloat[*input,1]
    prop = @network.transpose.dot(propagation)
    [
      Numo::SFloat[propagation.to_a].transpose.dot(Numo::SFloat[input.to_a]),
      Numo::SFloat[*prop.to_a.take(input.size-1)]
    ]
  end
end

class Loss2Layer
  attr_accessor :answer
  def initialize answer
    @answer = answer
  end

  def forward input
    ((input - answer) ** 2).sum
  end

  def backward input, propagation
    [0, 2 * (input - answer) * propagation]
  end
end

class ActivateLayerBase
  def forward input
    input.map { |x| activate x }
  end

  def backward input, propagation
    [0, input.map { |x| dactivate x } * propagation]
  end
end

class SigmoidLayer < ActivateLayerBase
  def activate x
    v = 1-1/(1+Math.exp(x))-0.5
    v.finite? ? v : (x < 0 ? 0 : 1)
  end

  def dactivate x
    v = Math.exp(x)/(1+Math.exp(x))**2
    v.finite? ? v : 0
  end
end

class SoftPlusLayer < ActivateLayerBase
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

class NN
  attr_reader :layers
  def initialize *layers, loss_layer_class: Loss2Layer
    @layers = layers
    @delta = 0.1
    @loss_layer_class = loss_layer_class
  end

  def forward input
    @inputs = @layers.map do |layer|
      layer_input = input
      input = layer.forward input
      layer_input
    end
    input
  end

  def backward propagation
    @layers.zip(@inputs).reverse_each.map { |layer, input|
      grad, propagation = layer.backward input, propagation
      grad
    }.reverse
  end

  def loss
    dataset = TrainingDataset.new
    yield dataset
    average_loss = dataset.map { |input, answer|
      @loss_layer_class.new(answer).forward forward(input)
    }.sum / dataset.size
  end

  def batch_train
    dataset = TrainingDataset.new
    yield dataset
    gradients = nil
    average_loss = dataset.map { |input, answer|
      loss_layer = @loss_layer_class.new(answer)
      output = forward(input)
      loss = loss_layer.forward output
      grad, propagation = loss_layer.backward output, 1
      grads = backward propagation
      if gradients
        gradients = gradients.zip(grads).map { |a, b| a+b }
      else
        gradients = grads
      end
      loss
    }.sum / dataset.size

    original_networks = @layers.map do |layer|
      layer.network if LinearLayer === layer
    end
    update = ->(delta){
      @layers.zip(original_networks, gradients).each do |layer, network, grad|
        next unless network
        layer.network = network - delta * grad
      end
      dataset.map{ |input, answer|
        @loss_layer_class.new(answer).forward forward(input)
      }.sum / dataset.size
    }
    10.times do
      updated_loss = update.call @delta
      if updated_loss < average_loss
        @delta *= 2
        return updated_loss
      end
      @delta /= 2
    end
  end

  class TrainingDataset < Array
    def learn input, output
      self << [input, output]
    end
  end
end

input = Numo::SFloat.new(1).seq

nn = NN.new(
  LinearLayer.new(2, 16),
  SigmoidLayer.new,
  LinearLayer.new(16, 1),
)

nn2 = NN.new(
  LinearLayer.new(2, 4),
  SoftPlusLayer.new,
  LinearLayer.new(4, 1)
)

nn2.layers[0].network = Numo::SFloat[[2,0,-1],[0,2,-1],[-2,0,1],[0,-2,1]]

nn2.layers[2].network = Numo::SFloat[[4, 4, 4, 4, -5]]

dataset = 10000.times.map{
  input = Numo::SFloat.new(2).rand
  [input, nn2.forward(input)]
}
p nn.loss{|d|dataset.each{|a|d<<a}}

require 'chunky_png'
img1 = ChunkyPNG::Image.new 128,128
img2 = ChunkyPNG::Image.new 128,128
def v2col a
  a=Math.sin(100*a)
  (((a<-1?-1:a>1?1:a)+1)/2*0xff).round
end
save=->{
  aaa=128.times.to_a.repeated_permutation(2).map{|i,j|
    x,y=i/128.0,j/128.0
    nn.forward(Numo::SFloat[x,y]).to_a
  }

  128.times.map{|i|128.times.map{|j|
    x,y=i/128.0,j/128.0
    c1, c2 = nn2.forward(Numo::SFloat[x,y]).to_a
    c2=c1
    img1[i,j] = (v2col(c1)<<16)|(v2col(c2)<<8)|0x000000ff
    c1, c2 = nn.forward(Numo::SFloat[x,y]).to_a
    c2=c1
    img2[i,j] = (v2col(c1)<<16)|(v2col(c2)<<8)|0x000000ff
  }}
  img1.save 'a.png'
  img2.save 'b.png'
}
train = ->{p nn.batch_train{|d|dataset.sample(100).each{|a|d<<a}}}
binding.pry
