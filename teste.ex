require PolyHok


Nx.tensor([[[1,2,3,4]]],type: {:s, 32})
|> PolyHok.new_gnx
|> PolyHok.get_gnx
|> IO.inspect