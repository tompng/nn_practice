require 'numo/narray'
require_relative 'layers/base'
require_relative 'layers/bias_layer'
require_relative 'layers/full_connected_layer'
require_relative 'layers/loss_layer'
require_relative 'layers/activate_layer'
require_relative 'layers/convolution_layer'


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
        gradients = gradients.zip(grads).map { |a, b| a + b }
      else
        gradients = grads
      end
      loss
    }.sum / dataset.size

    original_parameters = @layers.map &:parameter
    update = lambda do |delta|
      @layers.zip(original_parameters, gradients).each do |layer, params, grad|
        layer.update params, grad, delta
      end
      dataset.map{ |input, answer|
        @loss_layer_class.new(answer).forward forward(input)
      }.sum / dataset.size
    end
    @delta = [@delta, 1.0/(1<<6)].max
    4.times do
      updated_loss = update.call @delta
      if updated_loss <= average_loss
        @delta *= 1.5
        return updated_loss
      end
      @delta /= 4
    end
    update.call @delta
  end

  class TrainingDataset < Array
    def learn input, output
      self << [input, output]
    end
  end
end
