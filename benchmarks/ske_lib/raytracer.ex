require PolyHok

defmodule BMP do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/bmp_nifs', 0)
  end
  def gen_bmp_int_nif(_string,_dim,_mat) do
      raise "gen_bmp_nif not implemented"
  end
  def gen_bmp_float_nif(_string,_dim,_mat) do
    raise "gen_bmp_nif not implemented"
end
  def gen_bmp_int(string,dim,%Nx.Tensor{data: data, type: _type, shape: _shape, names: _name}) do
    %Nx.BinaryBackend{ state: array} = data
    gen_bmp_int_nif(string,dim,array)
  end
  def gen_bmp_float(string,dim,%Nx.Tensor{data: data, type: _type, shape: _shape, names: _name}) do
    %Nx.BinaryBackend{ state: array} = data
    gen_bmp_float_nif(string,dim,array)
  end
end

PolyHok.defmodule RayTracer do


defd raytracing(image, width,  spheres ,x,y) do

  ox = 0.0
  oy = 0.0
  ox = (x - width/2)
  oy = (y - width/2)

  r = 0.0
  g = 0.0
  b = 0.0

  maxz = -99999.0

  for i in range(0, 20) do

    sphereRadius = spheres[i * 7 + 3]

    dx = ox - spheres[i * 7 + 4]
    dy = oy - spheres[i * 7 + 5]
    n = 0.0
    t = -99999.0
    dz = 0.0
    if (dx * dx + dy * dy) <  (sphereRadius * sphereRadius) do
      dz = sqrtf(sphereRadius * sphereRadius - (dx * dx) - (dy * dy))
      n = dz / sqrtf(sphereRadius * sphereRadius)
      t = dz + spheres[i * 7 + 6]
    else
      t = -99999.0
      n = 0.0
    end

    if t > maxz do
      fscale = n
      r = spheres[i * 7 + 0] * fscale
      g = spheres[i * 7 + 1] * fscale
      b = spheres[i * 7 + 2] * fscale
      maxz = t
    end
  end

  image[0] = r * 255
  image[1] = g * 255
  image[2] = b * 255
  image[3] = 255

end

defk map_coord_2D_2para_no_resp_kernel(d_array,  step, par1, par2,sizex,sizey,f) do

  x = threadIdx.x + blockIdx.x * blockDim.x
  y = threadIdx.y + blockIdx.y * blockDim.y
  offset = x + y * blockDim.x * gridDim.x

   id  = step * offset
  #f(id,id)
  if (offset < (sizex*sizey)) do
    f(d_array+id,par1,par2,x,y)
  end
end
def map_coord_2D_2para_no_resp(d_array, par1, par2, f) do

  {sizex,sizey,step} =  case PolyHok.get_shape_gnx(d_array) do
                          {l,c} -> {l,c,1}
                          {l,c,step} -> {l,c,step}
                          x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                        end

  block_size = 16
  grid_rows = trunc ((sizex + block_size - 1) / block_size)
  grid_cols = trunc ((sizey + block_size - 1) / block_size)


  PolyHok.spawn(&RayTracer.map_coord_2D_2para_no_resp/6,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,step,par1,par2,sizex,sizey,f])
    d_array
end

end
defmodule Main do
    def rnd(x) do
        :rand.uniform() *x
        #x * Random.randint(1, 32767) / 32767
    end
    def sphereMaker2(0,_dim), do: []
    def sphereMaker2(n,dim) do
      [
        Main.rnd(1),
        Main.rnd(1),
        Main.rnd(1),
        Main.rnd(trunc(dim/10)) + (dim/50),
        Main.rnd( dim ) - trunc(dim/2),
        Main.rnd( dim ) - trunc(dim/2),
        Main.rnd( dim ) - trunc(dim/2)
        | sphereMaker2(n - 1,dim)]

    end

    def spherePrinter([]) do
      File.write!("spheregpu.txt", "done\n", [:append])

    end
    def spherePrinter([ r, g, b, _radius, _x, _y, _z | list]) do
      File.write!("spheregpu.txt", "\t r: #{r}", [:append])
      File.write!("spheregpu.txt", "\t g: #{g}", [:append])
      File.write!("spheregpu.txt", "\t b: #{b}", [:append])
      File.write!("spheregpu.txt", "\n", [:append])
      spherePrinter(list)
    end


    def sphereMaker(spheres, max, max) do
      max = max - 1
        Matrex.set(spheres, 1, max * 7 + 1, Main.rnd(1))
        |> Matrex.set( 1, max * 7 + 2, Main.rnd(1)) #g
        |> Matrex.set( 1, max * 7 + 3, Main.rnd(1)) #b
        |> Matrex.set( 1, max * 7 + 4, Main.rnd(20) + 5) #radius
        |> Matrex.set( 1, max * 7 + 5, Main.rnd(Main.dim) - Main.dim/2) #x
        |> Matrex.set( 1, max * 7 + 6, Main.rnd(Main.dim) - Main.dim/2) #y
        |> Matrex.set( 1, max * 7 + 7, Main.rnd(256) - 128) #z
    end
    def sphereMaker(spheres, n, max) do

      Matrex.set(spheres, 1, n * 7 + 1, Main.rnd(1)) #r
      |> Matrex.set( 1, (n - 1) * 7 + 2, Main.rnd(1)) #g
      |> Matrex.set( 1, (n - 1) * 7 + 3, Main.rnd(1)) #b
      |> Matrex.set( 1, (n - 1) * 7 + 4, Main.rnd(20) + 5) #radius
      |> Matrex.set( 1, (n - 1) * 7 + 5, Main.rnd(Main.dim) - Main.dim/2) #x
      |> Matrex.set( 1, (n - 1) * 7 + 6, Main.rnd(Main.dim) - Main.dim/2) #y
      |> Matrex.set( 1, (n - 1) * 7 + 7, Main.rnd(256) - 128) #z
      |> sphereMaker(n + 1, max)
    end

    def dim do
      {d, _} = Integer.parse(Enum.at(System.argv, 0))
      d
    end
    def spheres do
     # {s, _} = Integer.parse(Enum.at(System.argv, 1))
     # s
     20
    end

    def main do

        sphereList = Nx.tensor([sphereMaker2(Main.spheres,Main.dim)], type: {:f,32})

        width = Main.dim
        height = width



        prev = System.monotonic_time()

        ref_sphere = PolyHok.new_gnx(sphereList)
        ref_image = PolyHok.new_gnx(width, height, 4,{:s,32})

        Ske.map(ref_image, &RayTracer.raytracing/5 , [width, sphereList], dim: :two, return: false)
      
        _image = PolyHok.get_gnx(ref_image)

        next = System.monotonic_time()
        IO.puts "PolyHok\t#{width}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "


        #BMP.gen_bmp_int('ray.bmp',width,image)



    end
end

Main.main
