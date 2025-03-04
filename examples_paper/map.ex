require PolyHok

PolyHok.defmodule PMap do
  defk map_ker(a1,a2,size,f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(index,size,stride) do
          a2[id] = f(a1[id])
    end
  end
  defd inc(x) do
    x+1
  end
  def map(input, f) do
    shape = PolyHok.get_shape(input)
    type = PolyHok.get_type(input)
    result_gpu = PolyHok.new_gnx(shape,type)

    size = Tuple.product(shape)
    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(&PMap.map_ske/4,
              {numberOfBlocks,1,1},
              {threadsPerBlock,1,1},
              [input,result_gpu,size, f])
    result_gpu
  end

end

#a = Hok.hok (fn x,y -> x+y end)
#IO.inspect a
#raise "hell"

tensor1 = Nx.tensor([[1,2,3,4]],type: {:s, 32})
tensor2 = Nx.tensor([[1,2,3,4]],type: {:f, 32})
tensor3 = Nx.tensor([[1,2,3,4]],type: {:f, 64})
gtensor1 = Hok.new_gnx(tensor1)
gtensor2 = Hok.new_gnx(tensor2)
gtensor3 = Hok.new_gnx(tensor3)

func = Hok.hok fn (x) -> x + 1 end

#PMap.map(gtensor,&PMap.inc/1)

prev = System.monotonic_time()

gtensor1
    |> PMap.map()
    |> Hok.get_gnx
    |> IO.inspect

gtensor2
    |> PMap.map(func)
    |> Hok.get_gnx
    |> IO.inspect

gtensor3
    |> PMap.map(func)
    |> Hok.get_gnx
    |> IO.inspect

next = System.monotonic_time()
IO.puts "Hok\t\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"
