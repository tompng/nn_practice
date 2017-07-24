require 'pry'
require './nn'
input = Numo::SFloat.new(1).rand
fcin = FullConnectedLayer.new 1, 16
conv = SimpleConvolutionLayer.new w: 4, h: 4, size: 2
# conv = MaxPoolingLayer.new in_w: 4, in_h: 4, out_w: 2, out_h: 2, pool: 2
# conv = PaddingLayer.new w:3, h:3, padding:1
fcout = FullConnectedLayer.new 9, 1

i0 = input
i1 = fcin.forward i0
i2 = conv.forward i1
i3 = fcout.forward i2

pa0 = fcin.parameter
pa1 = conv.parameter
pa2 = fcout.parameter

reset = ->{
  fcin.update pa0,pa0,0
  conv.update pa1,pa1,0
  fcout.update pa2,pa2,0
}
test = ->{(fcout.forward conv.forward fcin.forward i0)[0]-i3[0]}
$g1 = fcin.parameter.size.times.map{|i|reset.call;fcin.parameter[i]+=0.001;test.call*1000}
$g2 = conv.parameter.size.times.map{|i|reset.call;conv.parameter[i]+=0.001;test.call*1000}

g3, p2 = fcout.backward i2, Numo::SFloat.new(1).fill(1)
g2, p1 = conv.backward i1, p2
g1, p0 = fcin.backward i0, p1

binding.pry
