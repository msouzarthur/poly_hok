require PolyHok

PolyHok.defmodule MCOV do

  defk mean_kernel(data) do
    data[0] = sqrt(data[0])
  end

  def run() do

    random_tensor = PolyHok.new_nx_from_function(1,1,{:f,32},fn -> :rand.uniform(100) end)
    |> IO.inspect()
    d_data = PolyHok.new_gnx(random_tensor)

    block_size = 16
    n_blocks = 16

    run_time = System.monotonic_time()
    # Chama o kernel para calcular a media
    PolyHok.spawn(
      &MCOV.mean_kernel/1,
      {n_blocks, 1, 1},
      {block_size, 1, 1},
      [d_data]
    )
    %{
      covariance: PolyHok.get_gnx(d_data)
    }
  end

end

result = MCOV.run()

IO.puts("covariance: #{inspect(result.covariance)}")
