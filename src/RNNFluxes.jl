#__precompile__()

module RNNFluxes

import Knet: Sgd, Momentum, Nesterov, Adagrad, Adadelta, Rmsprop, Adam

#include("update.jl")
include("plotProgress.jl")
include("train.jl")
include("RNN.jl")
include("LSTM.jl")
include("Generic.jl")
include("normalization.jl")
include("testdata.jl")


end # module
