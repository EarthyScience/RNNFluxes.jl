# RNNFluxes

[![Build Status](https://travis-ci.org/meggart/RNNFluxes.jl.svg?branch=master)](https://travis-ci.org/meggart/RNNFluxes.jl)

[![Coverage Status](https://coveralls.io/repos/meggart/RNNFluxes.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/meggart/RNNFluxes.jl?branch=master)

[![codecov.io](http://codecov.io/github/meggart/RNNFluxes.jl/coverage.svg?branch=master)](http://codecov.io/github/meggart/RNNFluxes.jl?branch=master)

# To-Do

* Make the LSTM Gradient pretty and fast (á la Fabian) DONE
* Implement truncated BPTT and Dropout
* Add GRU and/or LSTM with peepholes
* Implement more Loss functions
* Should some biases be set to zero?
* Implement unit tests
* Implement option to use autograd
* Implement getSample function

# Benchmarking: RNN vs LSTM

When both are using the old __autograd__ automatic diff to do the backprop, RNN is about __3__ times faster than LSTM.

RNN with the explicit gradient is about __28__ times faster than the LSTM with the auto-diff.

However, the LSTM seems to perform better - when epoch and sample size is high.

# Usage in IJulia Notebook

Load some traing data.

````julia
using Interact
using Plots
using RNNFluxes
default(show=:inline)
x,y = loadSeasonal(nSample=200)
x=[x[:,i,:]' for i=1:size(x,2)]
y=[y[:,i,:]' for i=1:size(y,2)];
````

Define some network parameters and construct an `RNNModel` while specifying the type of RNN:

````julia
nVarX = 3 # NUmber of input variables
nHid  = 30        # Number of hidden nodes
m     = RNNFluxes.LSTMModel(nVarX, nHid) # Initialize model shape and weights
````

Then we can start the straing. If we do this with an `async` macro, the plot updates automatically.

````julia
t = train_net(m,x,y,4001,batchSize=20,plotProgress=true,infoStepSize=100,searchParams=RNNFluxes.Adam(m.weights; lr=0.005, beta1=0.5, beta2=0.75, t=1, eps=1e-6, fstm=zeros(m.weights), scndm=zeros(m.weights)))
````
