require 'pry'
require_relative 'nn'
require_relative 'mnist'

mnist_train = MNIST.new 'data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte'

nn = NN.new(
  FullConnectedBiasedLayer.new(28*28, 64),
  ReLULayer.new,
  FullConnectedBiasedLayer.new(64, 10),
  ReLULayer.new,
  SoftmaxLayer.new
)

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
