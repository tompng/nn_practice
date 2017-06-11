require 'numo/narray'

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
    prop = propagation.dot @network
    [
      Numo::SFloat[propagation.to_a].transpose.dot(Numo::SFloat[input.to_a]),
      Numo::SFloat[*prop.to_a.take(input.size-1)]
    ]
  end
end

class SoftmaxLayer
  def forward input
    max = input.max
    exps = input.map { |v| Math.exp(v-max) }
    sum = exps.sum
    exps.map { |v| v / sum }
  end

  def backward input, propagation
    max = input.max
    exps = input.map { |v| Math.exp(v-max) }
    sum2 = exps.sum**2
    ep = exps * propagation
    [0, (ep + ep.sum * exps) / sum2]
  end
end

class CrossEntropyLossLayer
  attr_accessor :answer
  def initialize answer
    @answer = answer
  end

  def forward input
    -input.to_a.zip(@answer).map { |v, t| Math.log(v) * t }.sum
  end

  def backward input, propagation
    [0, Numo::SFloat.asarray(input.to_a.zip(@answer).map { |v, t| -t/(1e-10+v) } * propagation)]
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
    @delta = [@delta, 1.0/(1<<24)].max
    10.times do
      updated_loss = update.call @delta
      if updated_loss < average_loss
        @delta *= 2
        return updated_loss
      end
      @delta /= 2
    end
    p [:delta, @delta]
  end

  class TrainingDataset < Array
    def learn input, output
      self << [input, output]
    end
  end
end
