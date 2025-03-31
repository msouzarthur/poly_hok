require PolyHok
#PolyHok.defmodule SkeKernels do
#
# defk map_step_2_para_no_resp_kernel(d_array,  step, par1, par2,size,f) do
#    globalId  = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x
#    id  = step * globalId
#    #f(id,id)
#    if (globalId < size) do
#      f(d_array+id,par1,par2)
#    end
#  end
#
#end

PolyHok.defmodule Ske do
  #defmacro __using__(_opts) do
  #     IO.puts "You are USIng!"
  #    end
  @defaults %{coord: false, return: true, dim: :one}
  def map({:nx, type, shape, name , ref}, func, [par1,par2], options \\ [])do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   case shape do
                  {_m,_n} -> if (not coord && not return )do
                                map_2_para_no_resp({:nx, type, shape, name , ref},  par1, par2, func)
                              end
              end                
     :two ->  case shape do
                    {_m,_n,_o} -> if (coord && not return) do
                                      map_coord_2D_2para_no_resp({:nx, type, shape, name , ref}, par1, par2, func)
                                  end
              end
  end
  end
  def map(input, f) do
    shape = PolyHok.get_shape(input)
    type = PolyHok.get_type(input)
    result_gpu = PolyHok.new_gnx(shape,type)
    size = Tuple.product(shape)
    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(&Ske.map_ker/4,
              {numberOfBlocks,1,1},
              {threadsPerBlock,1,1},
              [input,result_gpu,size, f])
    result_gpu
  end
  defk map_coord_2D_2para_no_resp_kernel(d_array,  step, par1, par2,sizex,sizey,f) do

    x = threadIdx.x + blockIdx.x * blockDim.x
    y = threadIdx.y + blockIdx.y * blockDim.y
    offset = x + y * blockDim.x * gridDim.x
  
     id  = step * offset
    #f(id,id)
    if (offset < (sizex*sizey)) do
      f(d_array+id,par1,par2,x,y)
    end
  end
  def map_coord_2D_2para_no_resp(d_array, par1, par2, f) do
  
    {sizex,sizey,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
  
    block_size = 16
    grid_rows = trunc ((sizex + block_size - 1) / block_size)
    grid_cols = trunc ((sizey + block_size - 1) / block_size)
  
  
    PolyHok.spawn(&RayTracer.map_coord_2D_2para_no_resp/6,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,step,par1,par2,sizex,sizey,f])
      d_array
  end
  def map_2_para_no_resp(d_array,  par1, par2, f) do
      block_size =  128;
      {l,step} = PolyHok.get_shape_gnx(d_array)
      size = l*step
      nBlocks = floor ((size + block_size - 1) / block_size)

      PolyHok.spawn(&Ske.map_step_2_para_no_resp_kernel/6,{nBlocks,1,1},{block_size,1,1},[d_array,step,par1,par2,l,f])
      d_array
  end
  defk map_step_2_para_no_resp_kernel(d_array,  step, par1, par2,size,f) do
        globalId  = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x
        id  = step * globalId
        #f(id,id)
        if (globalId < size) do
          f(d_array+id,par1,par2)
        end
  end
  defk map_ker(a1,a2,size,f) do
      index = blockIdx.x * blockDim.x + threadIdx.x
      stride = blockDim.x * gridDim.x

      for i in range(index,size,stride) do
            a2[i] = f(a1[i])
      end
    end
  end





#### kernels
