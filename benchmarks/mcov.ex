defmodule MCOV do
  def transpose(matrix), do: Enum.zip(matrix) |> Enum.map(&Tuple.to_list/1)

  def mean(list), do: Enum.sum(list) / length(list)

  def demean_columns(matrix_data) do
    cols = transpose(matrix_data)
    means = Enum.map(cols, &mean/1)

    Enum.map(matrix_data, fn row ->
      Enum.zip(row, means)
      |> Enum.map(fn {x, m} -> x - m end)
    end)
  end

  def dot(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {x, y} -> x * y end)
    |> Enum.sum()
  end

  def generate_data(n_samples, n_var) do
    for _ <- 1..n_samples do
      for _ <- 1..n_var, do: :rand.uniform()
    end
  end

  def cov_matrix(n_samples, n_var) do
    matrix_data = generate_data(n_samples, n_var)
    # matrix_data = [
    #   [55.2, 545.1, 63.4],
    #   [45.1, 54.2, 65.3],
    #   [55.2, 545.1, 63.4]
    # ]


    covariance = demean_columns(matrix_data)
    vars = transpose(covariance)

    result = for xi <- vars do
      for xj <- vars do
        dot(xi, xj)/(n_samples-1)
      end
    end

    %{
      matrix_data: result
    }
  end
end

prev = System.monotonic_time()

[arg1, arg2] = System.argv()
n_samples = String.to_integer(arg1)
n_var = String.to_integer(arg2)

result = MCOV.cov_matrix(n_samples, n_var)

next = System.monotonic_time()
IO.puts("tempo: #{System.convert_time_unit(next-prev,:native,:millisecond)}ms")
IO.puts("dados: #{inspect(result.matrix_data)}")
