require PolyHok
PolyHok.defmodule ReturnsCalculator do
  include Random

  defk compute_returns(prices, returns, n_days, n_assets) do
    idx = blockIdx.x*blockDim.x+threadIdx.x

    if idx < n_assets do
      base = idx*n_days
      i = 1
      while i < n_days do
        returns[base+i-1] = (prices[base+i]-prices[base+i-1])/prices[base+i-1]
        i = i+1
      end
    end
  end

  defk apply_jumps(returns, jump_prob, jump_size, n) do
    idx = blockIdx.x*blockDim.x+threadIdx.x

    if idx < n do

      seed = idx+1.0*atomic_random()
      rand_val = 1.0*atomic_random()
      if rand_val < jump_prob do
        jump = (rand_val/jump_prob)*jump_size

        if rand_val > 0.5 do
          returns[idx] = returns[idx]+jump
        else
          returns[idx] = returns[idx]-jump
        end

      end
    end
  end

  defk adjust_liquidity(returns, volumes, liquidity_factors, n) do
    idx = blockIdx.x*blockDim.x+threadIdx.x

    if idx < n do
      log_volume = (volumes[idx]-1.0)-(volumes[idx]-1.0)*(volumes[idx]-1.0)/2.0
      liquidity_factors[idx] = 1.0/(1.0+0.1*abs(log_volume))
      returns[idx] = returns[idx]*liquidity_factors[idx]
    end
  end

  defk compute_var(returns, var_results, n_assets, n_days, coeficiente) do
    idx = blockIdx.x*blockDim.x+threadIdx.x

    if idx < n_assets do
      base = idx*(n_days-1)

      i = 0
      while i < n_days-2 do
        j = 0
        while j < n_days-i-2 do
          if returns[base+j] > returns[base+j+1] do
            temp = returns[base+j]
            returns[base+j] = returns[base+j+1]
            returns[base+j+1] = temp
          end
          j = j+1
        end
        i = i+1
      end

      var_index = trunc((n_days-1)*(1.0-coeficiente))
      var_results[idx] = returns[base + var_index]
    end
  end

  def calculate(asset_prices, volumes) do
    prices_tensor = Nx.tensor(asset_prices)
    {n_assets, n_days} = Nx.shape(prices_tensor)

    prices_buf = PolyHok.new_gnx(prices_tensor)
    liquidity_factors_buf = PolyHok.new_gnx(Nx.broadcast(0.0, {n_assets, n_days - 1}))
    returns_buf = PolyHok.new_gnx(Nx.broadcast(0.0, {n_assets, n_days - 1}))
    volumes_buf = PolyHok.new_gnx(volumes)
    var_results_buf = PolyHok.new_gnx(Nx.broadcast(0.0, {n_assets}))

    block_size = 256
    grid_size = div(n_assets+block_size-1, block_size)

    PolyHok.spawn(
      &ReturnsCalculator.compute_returns/4,
      {grid_size, 1, 1},
      {block_size, 1, 1},
      [prices_buf, returns_buf, n_days, n_assets]
    )

    PolyHok.spawn(
      &ReturnsCalculator.apply_jumps/4,
      {grid_size,1,1},
      {block_size,1,1},
      [returns_buf, 0.05, 0.15, n_assets*(n_days-1)]
    )

    PolyHok.spawn(
      &ReturnsCalculator.adjust_liquidity/4,
      {grid_size,1,1},
      {block_size,1,1},
      [returns_buf, volumes_buf, liquidity_factors_buf, n_assets*(n_days-1)]
    )

    PolyHok.spawn(
      &ReturnsCalculator.compute_var/5,
      {grid_size,1,1},
      {block_size,1,1},
      [returns_buf, var_results_buf, n_assets, n_days, 0.95]
    )

    PolyHok.get_gnx(returns_buf)
  end
end

prices = [
  [100.0, 101.5, 100.8, 102.3, 101.9],
  [50.0, 49.5, 48.8, 49.2, 50.1],
  [75.0, 76.2, 77.0, 76.5, 77.8]
]
volumes = Nx.tensor([
  [1.0, 1.2, 0.8, 1.5, 1.3],
  [2.0, 1.8, 2.1, 2.0, 2.2],
  [0.5, 0.6, 0.7, 0.6, 0.8]
])

returns = ReturnsCalculator.calculate(prices, volumes)

IO.inspect(returns)
