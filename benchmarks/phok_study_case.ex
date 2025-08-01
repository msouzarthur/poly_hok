require PolyHok

# PolyHok.defmodule SC do

#   def load_csv(path) do
#     File.stream!(path)
#     |> Enum.map(fn linha ->
#       [data | valores] = String.split(linha, ",")
#       Enum.map(valores, &String.to_float/1)
#     end)
#     |> PolyHok.new_gnx(:f32)  # Converte para tensor GPU
#   end

#   defk covariancia_massiva(retornos, cov, n_ativos, n_dias) do
#     i = blockIdx.x * 1024 + threadIdx.x  # Cada bloco cobre 1024 ativos
#     j = blockIdx.y * 1024 + threadIdx.y

#     if i < n_ativos and j <= i do  # Aproveita simetria
#       sum = 0.0
#       for dia <- 0..(n_dias-1) do
#         sum += retornos[i*n_dias + dia] * retornos[j*n_dias + dia]
#       end
#       cov[i*n_ativos + j] = sum / n_dias
#       cov[j*n_ativos + i] = cov[i*n_ativos + j]  # Espelha a matriz
#     end
#   end

#   defk qr_massivo(matriz, q, r, n_ativos, n_fatores) do
#     k = blockIdx.x  # Cada bloco = 1 fator (coluna)
#     i = threadIdx.x # 1024 threads por coluna

#     if k < n_fatores and i < n_ativos do
#       # Norma da coluna k (redução paralela)
#       r[k*n_fatores + k] = :math.sqrt(Enum.sum(for m <- 0..(n_ativos-1), do: matriz[m*n_fatores + k] ** 2))

#       # Ortogonalização
#       q[i*n_fatores + k] = matriz[i*n_fatores + k] / r[k*n_fatores + k]

#       # Atualiza colunas restantes
#       for j <- (k+1)..(n_fatores-1) do
#         r[k*n_fatores + j] = Enum.sum(for m <- 0..(n_ativos-1), do: q[m*n_fatores + k] * matriz[m*n_fatores + j])
#         matriz[i*n_fatores + j] -= q[i*n_fatores + k] * r[k*n_fatores + j]
#       end
#     end
#   end

#   defk fatorar_lu_massivo(a, l, u, n) do
#     k = blockIdx.x  # Fase k da fatoração
#     i = threadIdx.x # Linha i

#     if k < n and i < n do
#       if i >= k do
#         u[k*n + i] = a[k*n + i] - Enum.sum(for m <- 0..(k-1), do: l[k*n + m] * u[m*n + i])
#       end

#       if i > k do
#         l[i*n + k] = (a[i*n + k] - Enum.sum(for m <- 0..(k-1), do: l[i*n + m] * u[m*n + k])) / u[k*n + k]
#       end
#     end
#   end
# end

defmodule ReaderCSV do
  def read(path) do
    path
      |> Path.expand(__DIR__)
      |> File.stream!()
      |> Stream.map(&String.trim/1)
      |> Stream.map(&String.split(&1, ","))
      |> Enum.to_list()
      |> parse()
  end

  defp parse([header1, header2 | dados]) do
    cols = Enum.zip([header1, header2])
      |> Enum.map(fn {p, t} -> String.trim("#{p}_#{t}", "_") end)

    [first_col | title] = cols

    Enum.map(
      dados, fn [data | values] ->
        line = Enum.zip(title, values)
        Map.new([{first_col, data} | line])
      end
    )
  end
end

[arg] = System.argv()

# block_size = 512
# n_blocks = ceil(n_points/block_size)

# Leitura do arquivo csv com os dados das acoes
ReaderCSV.read(arg)
  |> Enum.take(5)
  |> IO.inspect()


# results = PolyHok.new_gnx({round(n_points)}, {:f, 64})

# prev = System.monotonic_time()

# hits = MC.mc(n_blocks, block_size, results, n_points)
#   |> MC.reduce(0.0,PolyHok.phok fn (a,b) -> a + b end)
#   |> PolyHok.get_gnx()
#   |> Nx.squeeze()
#   |> Nx.to_number()

# next = System.monotonic_time()

# IO.puts("""
#   tempo #{System.convert_time_unit(next-prev,:native,:millisecond)}ms
#   pontos #{n_points}
#   pi #{:io_lib.format("~.10f", [pi_estimate])}
#   erro #{:io_lib.format("~.6f", [abs(pi_estimate-:math.pi())/:math.pi()*100])}%
# """)
