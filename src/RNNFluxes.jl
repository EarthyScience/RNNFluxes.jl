__precompile__()

module RNNFluxes
using Interact

include("update.jl")
include("plotProgress.jl")
include("rnn.jl")
include("testdata.jl")

importall .PlotProgress
importall .RNN
importall .TestData

export train_net, predict_after_train, iniWeights, loadSeasonal, RNNModel



end # module
