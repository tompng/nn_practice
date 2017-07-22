require 'chunky_png'

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

  def self.to_img v, w: 28, h: 28
    img = ChunkyPNG::Image.new w, h
    h.times.to_a.product w.times.to_a do |i, j|
      c = (v[w*j + i]*0xff).round
      img[i, j] = (c*0x1010100) | 0xff
    end
    img
  end
end
