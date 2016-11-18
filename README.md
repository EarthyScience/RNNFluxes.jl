# RNNFluxes

[![Build Status](https://travis-ci.org/meggart/RNNFluxes.jl.svg?branch=master)](https://travis-ci.org/meggart/RNNFluxes.jl)

[![Coverage Status](https://coveralls.io/repos/meggart/RNNFluxes.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/meggart/RNNFluxes.jl?branch=master)

[![codecov.io](http://codecov.io/github/meggart/RNNFluxes.jl/coverage.svg?branch=master)](http://codecov.io/github/meggart/RNNFluxes.jl?branch=master)

# Usage

Load some traing data.

````julia
using RNNFluxes
x,y = loadSeasonal(nSample=200)
````

Define some network parameters and construct an `RNNModel`:

````julia
nVarX = size(x,3) # NUmber of input variables
nHid  = 12        # Number of hidden nodes
m     = RNNModel(3,12)
````

Then we can start the straing. If we do this with an `async` macro, the plot updates automatically.

````julia
t=@async train_net(m,x,y,501,batchSize=5,plotProgress=true, infoStepSize=20);
````

and predict on new values

````julia
predict_after_train(m,x)
````
