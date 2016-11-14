__precompile__()

module RNNFluxes
using Interact

include("update.jl")
include("plotProgress.jl")
include("rnn.jl")

importall .PlotProgress
importall .RNN

export train_net, predict_after_train, iniWeights



end # module
