
require PolyHok
PolyHok.defmodule MM do

 defk mm(a,b,c,m,n,k) do
  row  = blockIdx.y * blockDim.y + threadIdx.y
  col = blockIdx.x * blockDim.x + threadIdx.x
  sum  = 0.0
  if(col < k && row < m) do
    for i in range(0,n,1) do
      sum = sum + a[row * n + i] * b[i * k + col]
    end
    c[row * k + col] = sum
  end

end
end

[arg] = System.argv()

m = String.to_integer(arg)
n = m
k=m




mat = Matrex.fill(1,m*k,1)

f = fn _ -> Enum.random(1..100) end

mat1 = Matrex.apply(mat,f)
mat2 = Matrex.apply(mat,f)


block_size = 16
grid_rows = trunc ((m + block_size - 1) / block_size)
grid_cols = trunc ((k + block_size - 1) / block_size)

mat1 = PolyHok.new_nx_from_function(1,m*m,{:f,32},fn -> :rand.uniform(1000) end )
mat2 = PolyHok.new_nx_from_function(1,m*m,{:f,32},fn -> :rand.uniform(1000) end)

prev = System.monotonic_time()

a=PolyHok.new_gnx(mat1)
b=PolyHok.new_gnx(mat2)
c=PolyHok.new_gnx(1,m*k, {:f,32})

PolyHok.spawn(MM.mm/6,{grid_rows,grid_cols,1},{block_size,block_size,1},[a,b,c,m,n,k])

_result = PolyHok.get_gnx(c)

next = System.monotonic_time()
#IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"
IO.puts "PolyHok\t#{m}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "

#IO.inspect result
#IO.puts GPU.Backend.gen_c_kernel('addVectors',4,[])
