require 'pry'
require_relative 'nn'
require_relative 'mnist'

# nn = NN.new(
#   FullConnectedBiasedLayer.new(28*28, 64),
#   ReLULayer.new,
#   FullConnectedBiasedLayer.new(64, 10),
#   ReLULayer.new,
#   SoftmaxLayer.new
# )

nn = NN.new(
  VectorToSingleChannelLayer.new,
  MultiChannelConvolutionLayer.new(w: 28, h: 28, size: 2, insize: 1, outsize: 4),
  ChannelMapLayer.new(size: 4, layer: ReLULayer.new),
  ChannelMapLayer.new(size: 4, layer: MaxPoolingLayer.new(in_w: 27, in_h: 27, out_w: 13, out_h: 13, pool: 2)),
  MultiChannelConvolutionLayer.new(w: 13, h: 13, size: 2, insize: 4, outsize: 8),
  ChannelMapLayer.new(size: 8, layer: ReLULayer.new),
  ChannelMapLayer.new(size: 8, layer: MaxPoolingLayer.new(in_w: 12, in_h: 12, out_w: 6, out_h: 6, pool: 2)),
  MultiChannelConvolutionLayer.new(w: 6, h: 6, size: 3, insize: 8, outsize: 8),
  ChannelMapLayer.new(size: 8, layer: ReLULayer.new),
  ChannelMapLayer.new(size: 8, layer: MaxPoolingLayer.new(in_w: 4, in_h: 4, out_w: 2, out_h: 2, pool: 2)),
  ChannelConcatLayer.new,
  FullConnectedBiasedLayer.new(2*2*8, 32),
  ReLULayer.new,
  FullConnectedBiasedLayer.new(32, 10),
  # SoftmaxLayer.new
)

grad_test = ->delta=0.001{
  input = Numo::SFloat.new(28*28).rand
  layer = nn.instance_eval{@nn}
  output, cache = layer.forward_with_input_cache input
  grad, prop = layer.backward cache, Numo::SFloat.new(10).fill(1)
  param = layer.layers[1].instance_eval{@layer_matrix}[0][0].parameter
  param0 = param[0]
  param[0] += delta
  out2 = nn.forward input
  param[0] = param0
  [grad[1][0][0], (out2.sum - output.sum)/delta]
}



puts nn.parameter_size

mnist_train = MNIST.new 'data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte'

train = ->(batch_size=1024){
  loss = nn.batch_train do |d|
    batch_size.times do
      answer_label, dataset = mnist_train.sample
      answer = [0]*10
      answer[answer_label] = 1
      d.learn dataset, answer
    end
  end
  p loss
}

test = ->n=1000{
  correct = n.times.count{
    answer_label, dataset = mnist_train.sample
    output = nn.forward(dataset).to_a
    answer_label == output.index(output.max)
  }
  correct.fdiv n
}

show = ->{
  64.times{|i|
    v = nn.layers[0].network[i,0...28*28]
    min, max = v.minmax
    img = MNIST.to_img v.map{|v|(v-min)/(max-min)}
    img.save "out/tmp#{i}.png"
  }
}

binding.pry

100000000.times{|i|
  64.times{train.call 256}
  show.call
  p test.call
}
