require 'numo/narray'
require_relative 'layers/base'
require_relative 'layers/bias_layer'
require_relative 'layers/full_connected_layer'
require_relative 'layers/loss_layer'
require_relative 'layers/channel_layer'
require_relative 'layers/activate_layer'
require_relative 'layers/convolution_layer'

class NN
  def initialize *layers, loss_layer_class: Loss2Layer
    @nn = CompositeLayer.new layers
    @delta = 0.1
    @loss_layer_class = loss_layer_class
  end

  def layers
    @nn.layers
  end

  def parameter_size
    @nn.parameter.flatten.compact.map(&:to_a).flatten.size
  end

  def forward input
    output, @input_was = @nn.forward_with_input_cache input
    output
  end

  def backward propagation
    @nn.backward(@input_was, propagation).first
  end

  def loss
    dataset = TrainingDataset.new
    yield dataset
    dataset.map { |input, answer|
      @loss_layer_class.new(answer).forward forward(input)
    }.sum / dataset.size
  end

  def batch_train
    dataset = TrainingDataset.new
    yield dataset
    gradient = nil
    average_loss = 0
    dataset.map do |input, answer|
      loss_layer = @loss_layer_class.new(answer)
      output = forward(input)
      loss = loss_layer.forward output
      _grad, propagation = loss_layer.backward output, 1
      grad = backward propagation
      average_loss += loss / dataset.size
      gradient = gradient ? gradient + grad : grad if grad
    end
    original_parameter = @nn.parameter
    update = lambda do |delta|
      @nn.update original_parameter, gradient, delta
      dataset.map { |input, answer|
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
