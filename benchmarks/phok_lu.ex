require PolyHok

PolyHok.defmodule LU do
  @doc """
    Kernel que realiza o calculo da decomposicao LU de uma matriz

    ## Parametros
    - `data`: matriz de dados (Nx.Tensor) com as amostras
    - `l`: matriz L (Nx.Tensor) a ser preenchida
    - `u`: matriz U (Nx.Tensor) a ser preenchida
    - `n`: tamanho da matriz (inteiro)
  """
  defk lu_kernel(data, l, u, n) do
    idx = threadIdx.x + blockDim.x * blockIdx.x

    if idx < n do
      for k in range(0, n) do
        soma = 0.0
        if idx >= k do
          for j in range(0, k) do
            soma = soma+(l[k*n+j]*u[j*n+idx])
          end

          u[k*n+idx] = data[k*n+idx]-soma
        end

        if idx >= k do
          if idx == k do
            l[idx*n+k] = 1.0
          else
            soma = 0.0

            for j in range(0, k) do
              soma = soma+(l[idx*n+j]*u[j*n+k])
            end

            l[idx*n+k] = (data[idx*n+k]-soma)/u[k*n+k]
          end

        end

        __syncthreads()
      end

    end

  end
  @doc """
    Gera uma matriz aleatoria
    Executa o kernel de decomposicao LU usando a matriz gerada

    ## Parametros
    - `n_rows`: tamanho da matriz (inteiro)

    ## Retorno
    - Um mapa contendo o tempo de execucao e as matrizes L e U resultantes
    - - `time_ms`: tempo de execucao em milissegundos
    - - `matrix`: matriz original
    - - `matrix_l`: matriz L resultante
    - - `matrix_u`: matriz U resultante
  """
  def run(n_rows) do

    # {random_tensor, _} = Nx.Random.uniform(
    #   Nx.Random.key(:erlang.system_time()),
    #   shape: {n_rows, n_rows},
    #   type: :f32
    # )

    random_tensor = PolyHok.new_nx_from_function(n_rows,n_rows,{:f,32},fn -> :rand.uniform(100) end)

    # Aumenta os valores para facilitar a visualizacao
    random_tensor = random_tensor
      |> Nx.multiply(10)
      |> Nx.round()

    {d_matrix, d_l, d_u} = {
      PolyHok.new_gnx(random_tensor),
      PolyHok.new_gnx(Nx.broadcast(0.0, {n_rows, n_rows})),
      PolyHok.new_gnx(Nx.broadcast(0.0, {n_rows, n_rows}))
    }

    block_size = 128
    n_blocks = div(n_rows+block_size-1, block_size)

    run_time = System.monotonic_time()

    PolyHok.spawn(
      &LU.lu_kernel/4,
      {n_blocks, 1, 1},
      {block_size, 1, 1},
      [d_matrix, d_l, d_u, n_rows]
    )

    end_run_time = System.monotonic_time()

    %{
      run_time: System.convert_time_unit(end_run_time - run_time, :native, :millisecond),
      matrix: PolyHok.get_gnx(d_matrix),
      matrix_l: PolyHok.get_gnx(d_l),
      matrix_u: PolyHok.get_gnx(d_u)
    }
  end

end

prev = System.monotonic_time()

[arg1] = System.argv()
n_rows = String.to_integer(arg1)

result = LU.run(n_rows)

next = System.monotonic_time()

IO.puts("tempo: #{System.convert_time_unit(next-prev,:native,:millisecond)}ms")
IO.puts("tempo de execucao: #{result.run_time}ms")
# IO.inspect(result.matrix, label: "matriz")
# IO.inspect(result.matrix_l, label: "matriz L")
# IO.inspect(result.matrix_u, label: "matriz U")
