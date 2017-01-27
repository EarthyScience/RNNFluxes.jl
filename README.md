# RNNFluxes

[![Build Status](https://travis-ci.org/meggart/RNNFluxes.jl.svg?branch=master)](https://travis-ci.org/meggart/RNNFluxes.jl)

[![Coverage Status](https://coveralls.io/repos/meggart/RNNFluxes.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/meggart/RNNFluxes.jl?branch=master)

[![codecov.io](http://codecov.io/github/meggart/RNNFluxes.jl/coverage.svg?branch=master)](http://codecov.io/github/meggart/RNNFluxes.jl?branch=master)

# Benchmarking: RNN vs LSTM

When both are using the old __autograd__ automatic diff to do the backprop, RNN is about __3__ times faster than LSTM.

RNN with the explicit gradient is about __28__ times faster than the LSTM with the auto-diff.

However, the LSTM seems to perform better - when epoch and sample size is high.

# Usage

Load some traing data.

````julia
using RNNFluxes
x,y = loadSeasonal(nSample=200)
````

Define some network parameters and construct an `RNNModel` while specifying the type of RNN:

````julia
nVarX = size(x,3) # NUmber of input variables
nHid  = 12        # Number of hidden nodes
m     = RNNModel(3,12, "LSTM") # RNN or LSTM?
````

Then we can start the straing. If we do this with an `async` macro, the plot updates automatically.

````julia
@time train_net(m,x,y,501,batchSize=5,plotProgress=false, infoStepSize=20);
````

and predict on new values

````julia
predict_after_train(m,x)
````
