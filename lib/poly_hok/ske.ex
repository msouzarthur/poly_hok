require PolyHok

defmodule Ske do
  @defaults %{coord: false, return: true, dim: :one}
   def map({:nx, type, shape, name , ref}, func, [par1,par2], options \\ [])do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
     if (not coord && not return && dim == :one)do
      map_2_para_no_resp({:nx, type, shape, name , ref},  par1, par2, func)
     end
    end
     def map_2_para_no_resp(d_array,  par1, par2, f) do
      block_size =  128;
      {l,step} = PolyHok.get_shape_gnx(d_array)
      size = l*step
      nBlocks = floor ((size + block_size - 1) / block_size)

        PolyHok.spawn(&SkeKernels.map_step_2_para_no_resp_kernel/6,{nBlocks,1,1},{block_size,1,1},[d_array,step,par1,par2,l,f])
        d_array
    end
  end
PolyHok.defmodule SkeKernels do
  defk map_step_2_para_no_resp_kernel(d_array,  step, par1, par2,size,f) do
    globalId  = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x
    id  = step * globalId
    #f(id,id)
    if (globalId < size) do
      f(d_array+id,par1,par2)
    end
  end

end




#### kernels
