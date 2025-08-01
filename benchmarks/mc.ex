defmodule MC do

  def mc(n_points) do

    hits = Enum.reduce(1..n_points, 0, fn _, acc ->
        x = :rand.uniform()
        y = :rand.uniform()
        if x * x + y * y <= 1, do: acc + 1, else: acc
      end)

    hits
  end

end

prev = System.monotonic_time()

[arg] = System.argv()
n_points = String.to_integer(arg)

hits = MC.mc(n_points)

next = System.monotonic_time()

pi_estimate = 4.0 * hits / n_points

IO.puts("""
pi: #{:io_lib.format("~.10f", [pi_estimate])}
erro: #{:io_lib.format("~.6f", [abs(pi_estimate-:math.pi())/:math.pi()*100])}%
tempo: #{System.convert_time_unit(next-prev,:native,:millisecond)}ms
""")
