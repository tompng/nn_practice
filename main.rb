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
  MultiChannelConvolutionLayer.new(w: 28, h: 28, size: 3, insize: 1, outsize: 4),
  ChannelMapLayer.new(size: 4, layer: MaxPoolingLayer.new(in_w: 26, in_h: 26, out_w: 13, out_h: 13, pool: 2)),
  ChannelMapLayer.new(size: 4, layer: ReLULayer.new),
  MultiChannelConvolutionLayer.new(w: 13, h: 13, size: 3, insize: 4, outsize: 8),
  ChannelMapLayer.new(size: 8, layer: MaxPoolingLayer.new(in_w: 11, in_h: 11, out_w: 5, out_h: 5, pool: 2)),
  ChannelMapLayer.new(size: 8, layer: ReLULayer.new),
  MultiChannelConvolutionLayer.new(w: 5, h: 5, size: 3, insize: 8, outsize: 16),
  ChannelMapLayer.new(size: 16, layer: MaxPoolingLayer.new(in_w: 3, in_h: 3, out_w: 1, out_h: 1, pool: 3)),
  ChannelConcatLayer.new,
  FullConnectedBiasedLayer.new(16, 10),
  ReLULayer.new,
  SoftmaxLayer.new
)

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
