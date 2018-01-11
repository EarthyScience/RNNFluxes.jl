export loadSeasonal
import CSV

function loadSeasonal(;nSample=typemax(Int))
  df = CSV.read(joinpath(dirname(@__FILE__),"..","data","MLTestSeasonal2.csv"), header=true)

  xOrig = convert(Array, df[[Symbol("x.1"), Symbol("x.2"), Symbol("x.3")]])
  # Predictor has dimensions (time, sample, variables)
  xOrig = reshape(xOrig, 100, 500, 3)
  x = xOrig[:,1:min(nSample,500),:] ## For performance only choose fewer samples (sites)

  yOrig = convert(Array, df[:yOBS])
  yOrig = reshape(yOrig, 100, 500, 1)
  y = yOrig[:, 1:min(nSample,500), :] ## For performance only choose fewer samples (sites)
  x,y
end
