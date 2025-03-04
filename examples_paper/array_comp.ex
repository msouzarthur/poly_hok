host_array =  tensor1 = Nx.tensor([[1,2,3,4]],type: {:s, 32})

host_resp = PolyHok.gpu_for n <- host_array,  do: n * n
