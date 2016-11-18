module TestData
export loadSeasonal
using DataFrames, DataArrays

function loadSeasonal(;nSample=typemax(Int))
  df = readtable(joinpath(dirname(@__FILE__),"..","data","MLTestSeasonal2.csv"), header=true)

  xOrig = convert(Array, df[[:x_1, :x_2, :x_3]])
  # Predictor has dimensions (time, sample, variables)
  xOrig = reshape(xOrig, 100, 500, 3)
  x = xOrig[:,1:min(nSample,500),:] ## For performance only choose fewer samples (sites)

  yOrig = convert(Array, df[:yOBS])
  yOrig = reshape(yOrig, 100, 500, 1)
  y = yOrig[:, 1:min(nSample,500), :] ## For performance only choose fewer samples (sites)
  x,y
end
end
