defmodule LU do

  def lu(l, u, n) do
    {l, u} = for k <- 0..n-2, reduce: {l, u} do
      {l, u} ->
        for i <- k+1..n-1, reduce: {l, u} do
          {l, u} ->
            factor = Enum.at(Enum.at(u, i), k)/Enum.at(Enum.at(u, k), k)
            l = put_in(l, [Access.at(i), Access.at(k)], factor)
            u = for j <- k..n-1, reduce: u do
              u -> put_in(u, [Access.at(i), Access.at(j)], Enum.at(Enum.at(u, i), j)-factor*Enum.at(Enum.at(u, k), j))
            end
            {l, u}
        end
    end

    {l, u}
  end

  def gen_matrix(n_rows) do
    for i <- 0..n_rows-1 do
      for j <- 0..n_rows-1 do
        if i == j, do: 1.0, else: 0.0
      end
    end
  end
  def run(n_rows) do

    {random_tensor, _} = Nx.Random.uniform(
      Nx.Random.key(62),
      shape: {n_rows, n_rows},
      type: :f32
    )

    matrix = random_tensor
      |> Nx.multiply(10)
      |> Nx.round()
      |> Nx.to_list()

    u = matrix

    l = gen_matrix(n_rows)

    {l, u} = lu(l, u, n_rows)

    %{
      matrix: matrix,
      matrix_l: l,
      matrix_u: u
    }
  end

end

prev = System.monotonic_time()

[arg1] = System.argv()
n_rows = String.to_integer(arg1)

_result = LU.run(n_rows)

next = System.monotonic_time()

IO.puts("tempo: #{System.convert_time_unit(next-prev,:native,:millisecond)}ms")
# IO.inspect(result.matrix, label: "matriz")
# IO.inspect(result.matrix_l, label: "matriz l")
# IO.inspect(result.matrix_u, label: "matriz u")
