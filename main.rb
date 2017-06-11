require 'pry'
require_relative 'nn'
require 'chunky_png'
input = Numo::SFloat.new(1).seq

class MNIST
  def initialize image_file, label_file
    image_data = File.binread(image_file).bytes
    label_data = File.binread(label_file).bytes
    label_unknown = parse_bytes label_data[0,4]
    image_unknown = parse_bytes image_data[0,4]
    label_size = parse_bytes label_data[4,4]
    image_size = parse_bytes image_data[4,4]
    raise if label_size != image_size
    @size = image_size
    @w = parse_bytes image_data[8,4]
    @h = parse_bytes image_data[12,4]
    @datasets = @size.times.map do |i|
      Numo::SFloat.asarray(image_data[16+@w*@h*i, @w*@h]).map{|a|a/0xff}
    end
    @labels = @size.times.map { |i| label_data[8+i] }
  end

  def parse_bytes s
    s.reverse.each_with_index.map{|n,i|n.ord<<(8*i)}.sum
  end

  def [] i
    [@labels[i], @datasets[i]]
  end

  def sample
    if block_given?
      loop do
        a,b=self[rand @size]
        return a, b if yield a
      end
    end
    self[rand @size]
  end

  def to_img v
    img = ChunkyPNG::Image.new @w, @h
    @h.times.to_a.product @w.times.to_a do |i, j|
      c = (v[@w*j + i]*0xff).round
      img[i, j] = (c*0x1010100) | 0xff
    end
    img
  end
end

mnist_train = MNIST.new 'data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte'
mnist_train.to_img(mnist_train.sample.last).save('tmp.png')

nn = NN.new(
  LinearLayer.new(28*28, 64),
  SigmoidLayer.new,
  # LinearLayer.new(64, 64),
  # SigmoidLayer.new,
  LinearLayer.new(64, 10)
  # SoftmaxLayer.new,
  # loss_layer_class: CrossEntropyLossLayer
)

def smooth28 m
  m2 = m.dup
  28.times{|i|28.times{|j|
    m2[28*i+j] = (
      4*m[28*i+j]+
      (i>0 ? m[(i-1)*28+j] : m[i*28+j])+
      (j>0 ? m[i*28+j-1] : m[i*28+j])+
      (i<28-1 ? m[(i+1)*28+j] : m[i*28+j])+
      (j<28-1 ? m[i*28+j+1] : m[i*28+j])
    )/8
  }}
  m2
end

smoothlayer = ->{
  nw = nn.layers[0].network
  64.times{|i|nw[i,0...28*28]=smooth28(nw[i,0...28*28].to_a)}
}

train = ->(batch_size=128){
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

test = ->file=nil{
  idx = rand mnist_train.instance_eval{@size}
  answer_label, dataset = mnist_train[idx]
  mnist_train.to_img(dataset).save file if file
  # output = SoftmaxLayer.new.forward
  output = nn.forward(dataset).to_a
  result = output.each.with_index(0).to_a.sort_by(&:first).reverse
  p answer: answer_label, result: result.first.last, idx: idx
  puts result.map{|a|a.join(' ')}
}
test2 = ->{
  testsample = ->{
    answer_label, dataset = mnist_train.sample
    output = nn.forward(dataset).to_a
    [answer_label, output.index(output.max)]
  }
  a=1000.times.map{testsample.call}.group_by(&:itself).map{|a,b|[a,b.size]}.sort_by(&:last).reverse.to_h
  p 10.times.map{|i|a[[i,i]]||0}.sum
  a
}

show = ->{
  64.times{|i|
    v = nn.layers[0].network[i,0...28*28]
    p v.size
    min, max = v.minmax
    img = mnist_train.to_img v.map{|v|(v-min)/(max-min)}
    img.save "tmp#{i}.png"
  }
}

binding.pry
