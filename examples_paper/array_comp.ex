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

    PolyHok.spawn(&PMap.map_ker/4,
              {numberOfBlocks,1,1},
              {threadsPerBlock,1,1},
              [input,result_gpu,size, f])
    result_gpu
  end

end

host_array = Nx.tensor([[1,2,3,4]],type: {:s, 32})

host_resp = PolyHok.gpu_for n <- host_array,  do: n * n

IO.inspect host_resp
