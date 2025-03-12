require PolyHok
require PolyHok.Ske

n = 1000

arr1 = Nx.tensor([Enum.to_list(1..n)],type: {:s, 32})
arr2 = Nx.tensor([Enum.to_list(1..n)],type: {:f, 32})
arr3 = Nx.tensor([Enum.to_list(1..n)],type: {:f, 64})

host_res1 = arr1
    |> PolyHok.new_gnx
    |> PolyHok.Ske.map(PolyHok.phok fn (x) -> x + 1 end)
    |> PolyHok.get_gnx