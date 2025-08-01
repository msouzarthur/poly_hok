require PolyHok

PolyHok.defmodule MCOV do
  @doc """
    Kernel que realiza o calculo da media de cada linha da matriz
    Media eh usada no calculo da matriz de covariancia

    ## Parametros
    - `data`: matriz de dados (Nx.Tensor) com as amostras
    - `mean`: vetor de medias (Nx.Tensor) para a matriz de medias
    - `n_samples`: numero de amostras (inteiros) para a matriz
    - `n_var`: numero de variaveis (inteiros) para a matriz
  """
  defk mean_kernel(data, mean, n_samples, n_var) do
    j = blockIdx.x*blockDim.x+threadIdx.x

    if j < n_var do
      sum = 0.0
      k = 0
      while k < n_samples do
        sum = sum+data[k*n_var+j]
        k = k+1
      end

      mean[j] = sum/n_samples

    end

  end
  @doc """
    Kernel que realiza o calculo da matriz de covariancia

    ## Parametros
    - `data`: matriz de dados (Nx.Tensor) com as amostras
    - `mean`: vetor de medias (Nx.Tensor) para a matriz de medias
    - `cov`: matriz de covariancia (Nx.Tensor) a ser preenchida
    - `n_samples`: numero de amostras (inteiros) para a matriz
    - `n_var`: numero de variaveis (inteiros) para a matriz
  """
  defk cov_kernel(data, mean, cov, n_samples, n_var) do
    __shared__ shared_mean[512]

    i = blockIdx.x
    j = threadIdx.x

    if j < n_var do
      shared_mean[j] = mean[j]
    end

    __syncthreads()

    if i < n_var && j < n_var do
      sum = 0.0
      mean_i = shared_mean[i]
      mean_j = shared_mean[j]

      k = 0
      while k < n_samples do
        val_i = data[k*n_var+i]
        val_j = data[k*n_var+j]
        sum = sum+(val_i-mean_i)*(val_j-mean_j)
        k = k+1
      end

      cov[i*n_var+j] = sum/(n_samples-1)
    end

  end
  @doc """
    Prepara as matriz e aloca os enderecos de memoria necessarios
    Utiliza do Nx.Random para gerar uma matriz aleatoria

    ## Parametros
    - `n_samples`: numero de amostras (inteiros) para a matriz
    - `n_var`: numero de variaveis (inteiros) para a matriz

    ## Exemplos
        iex> MCOV.run(2, 3)
        iex> MCOV.run(500, 12)
  """
  def run(n_samples, n_var) do

    # Gera um vetor aleatorio de amostras
    # {random_tensor, _} = Nx.Random.uniform(
    #   Nx.Random.key(:erlang.system_time()),
    #   shape: {n_samples, n_var},
    #   type: :f32
    # )

    random_tensor = PolyHok.new_nx_from_function(n_samples,n_var,{:f,32},fn -> :rand.uniform(100) end)

    # Cria os tensores gpu para os dados, media e covariancia
    {d_data, d_mean, d_cov} = {
      PolyHok.new_gnx(random_tensor),
      PolyHok.new_gnx(Nx.broadcast(0.0, {n_var})),
      PolyHok.new_gnx(Nx.broadcast(0.0, {n_var, n_var}))
    }
    # {d_mean, d_cov} = {
    #     PolyHok.new_gnx(Nx.broadcast(0.0, {n_var})),
    #     PolyHok.new_gnx(Nx.broadcast(0.0, {n_var, n_var}))
    # }
    # d_data = PolyHok.new_gnx(Nx.tensor([
    #   [55.2, 545.1, 63.4],
    #   [45.1, 54.2, 65.3],
    #   [55.2, 545.1, 63.4]
    # ]))
    block_size = 256
    n_blocks = round((n_var+block_size-1)/block_size)

    # Inicia o tempo
    run_time = System.monotonic_time()
    # Chama o kernel para calcular a media
    PolyHok.spawn(
      &MCOV.mean_kernel/4,
      {n_blocks, 1, 1},
      {block_size, 1, 1},
      [d_data, d_mean, n_samples, n_var]
    )
    # Chama o kernel para calcular a covariancia
    PolyHok.spawn(
      &MCOV.cov_kernel/5,
      {n_var, 1, 1},
      {n_var, 1, 1},
      [d_data, d_mean, d_cov, n_samples, n_var]
    )
    # Encerra o calculo do tempo
    end_run_time = System.monotonic_time()

    result = PolyHok.get_gnx(d_cov)
    # mean = PolyHok.get_gnx(d_mean)
    # data = PolyHok.get_gnx(d_data)

    %{
      run_time: System.convert_time_unit(end_run_time - run_time, :native, :millisecond),
      covariance: result
    }
  end

end

prev = System.monotonic_time()

[arg1, arg2] = System.argv()
n_samples = String.to_integer(arg1)
n_var = String.to_integer(arg2)

result = MCOV.run(n_samples, n_var)

next = System.monotonic_time()

IO.puts("tempo: #{System.convert_time_unit(next-prev,:native,:millisecond)}ms")
IO.puts("tempo de execucao: #{result.run_time}ms")
# IO.puts("covariance: #{inspect(result.covariance)}")
