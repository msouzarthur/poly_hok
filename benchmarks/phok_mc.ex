require PolyHok

PolyHok.defmodule MC do
  include CAS_Double
  include Random

  def mc(n_blocks, block_size, results, n_points) do

    PolyHok.spawn(
      &MC.gpu_mc/2,
      {n_blocks, 1, 1},
      {block_size, 1, 1},
      [results, n_points]
    )

    results
  end

  defk gpu_mc(results, n_points) do

    idx = blockIdx.x * blockDim.x + threadIdx.x
    count = 0.0

    x = atomic_random()+0.0
    y = atomic_random()+0.0

    if x*x + y*y <= 1.0 do
      count = count + 1.0
    end

    results[idx] = count
  end

  def reduce(ref, initial, f) do

    {l} = PolyHok.get_shape_gnx(ref)
    type = PolyHok.get_type_gnx(ref)
    size = l
    result_gpu  = PolyHok.new_gnx(Nx.tensor([[initial]] , type: type))

    threadsPerBlock = 512
    blocksPerGrid = div(size + threadsPerBlock - 1, threadsPerBlock)
    numberOfBlocks = blocksPerGrid

    PolyHok.spawn(
      &MC.reduce_kernel/5,
      {numberOfBlocks,1,1},
      {threadsPerBlock,1,1},
      [ref,result_gpu, initial,f, size]
    )

    result_gpu
  end

  defk reduce_kernel(a, ref4, initial, f, n) do
    __shared__ cache[512]

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    cacheIndex = threadIdx.x

    temp = initial+0.0

    while (tid < n) do
      temp = f(a[tid], temp)
      tid = blockDim.x * gridDim.x + tid
    end

    cache[cacheIndex] = temp
      __syncthreads()

    i = blockDim.x/2

    while (i != 0 ) do
      if (cacheIndex < i) do
        cache[cacheIndex] = f(cache[cacheIndex + i] , cache[cacheIndex])
      end

      __syncthreads()
      i = i/2
    end

    if (cacheIndex == 0) do
      current_value = ref4[0]
      while(!(current_value == atomic_cas(ref4, current_value, f(cache[0], current_value)))) do
      current_value = ref4[0]
      end

    end

  end

  def run(n_points) do
    block_size = 512
    n_blocks = ceil(n_points/block_size)

    results = PolyHok.new_gnx({round(n_points)}, {:f, 64})

    run_time = System.monotonic_time()

    hits = MC.mc(n_blocks, block_size, results, n_points)
      |> MC.reduce(0.0,PolyHok.phok fn (a,b) -> a + b end)
      |> PolyHok.get_gnx()
      |> Nx.squeeze()
      |> Nx.to_number()

    end_run_time = System.monotonic_time()

    pi_estimate = 4.0 * hits / n_points

    %{
      pi: pi_estimate,
      run_time: System.convert_time_unit(end_run_time - run_time, :native, :millisecond),
    }
  end

end

prev = System.monotonic_time()

[arg] = System.argv()
n_points = String.to_integer(arg)

result = MC.run(n_points)

next = System.monotonic_time()

IO.puts("tempo: #{System.convert_time_unit(next-prev,:native,:millisecond)}ms")
IO.puts("tempo de execucao: #{result.run_time}ms")
# IO.puts("pi: #{:io_lib.format("~.10f", [result.pi])}")
# IO.puts("erro: #{:io_lib.format("~.6f", [abs(result.pi-:math.pi())/:math.pi()*100])}%")
