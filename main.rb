require 'pry'
require_relative 'nn'
input = Numo::SFloat.new(1).seq

nn = NN.new(
  LinearLayer.new(2, 16),
  SigmoidLayer.new,
  LinearLayer.new(16, 2),
)

def answer v
  x, y = v.to_a
  v = (x-0.5)**2+(y-0.5)**2 < 0.2 ? 1 : 0
  Numo::SFloat[v, 1-v]
end

dataset = 10000.times.map{
  input = Numo::SFloat.new(2).rand
  [input, answer(input)]
}
p nn.loss{|d|dataset.each{|a|d<<a}}

require 'chunky_png'
def saveimg file
  img = ChunkyPNG::Image.new 128,128
  v2col = ->a{
    a=(a*10).round*0.1
    ((a<0?0:a>1?1:a)*0xff).round
  }
  128.times.map{|i|128.times.map{|j|
    x,y=i/128.0,j/128.0
    c1, c2 = yield(x, y).to_a
    img[i,j] = (v2col[c1]<<16)|(v2col[c2]<<8)|0x000000ff
  }}
  img.save file
end
saveimg('a.png'){|x,y|answer Numo::SFloat[x,y]}
save = ->{saveimg('b.png'){|x,y|SoftmaxLayer.new.forward nn.forward Numo::SFloat[x,y]}}

train = ->{p nn.batch_train{|d|dataset.sample(100).each{|a|d<<a}}}
binding.pry
