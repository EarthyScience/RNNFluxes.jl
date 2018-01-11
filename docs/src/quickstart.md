# A quick Overview

Here is a quick demonstration of how to train and apply an LSTM. Lets create some artificial input data first:

````julia
nTime=100
nSample=200
nVarX = 3
x = [rand(nVarX,nTime) for i=1:nSample]

true_model(x) = transpose(cumsum(x[1,:])+exp.(x[2,:])+cumprod(1.7*x[3,:]))

y = true_model.(x);
````

Let's see what we have here. x contains the predictor variables. It is contains 200 samples of multivariate time series. Each sample is a `nVar` x `nTime` matrix containing the inputs for the model. Our function `true_model` generates a new time series from such a multivariate time series and is applied to every input sample.
So our target variable y contains `nSample` row Vectors of length `ntime`.

Now we create a model:

````julia
using RNNFluxes
nHid=15
m=RNNFluxes.LSTMModel(nVarX,nHid,nDropout=1)
````

This initializes an LSTM model containing 15 hidden nodes which represent the persistent state of the model. To train the model we can run:

````julia
train_net(m,x,y,5001);
````

This trains the model m for 5001 update steps. Note that the model variable `m` is updated in-place. To test how well our model performs,
we generate a test dataset:

````julia
xtest = [rand(nVarX,nTime)]
ytest = true_model.(xtest)[1]
````

and do a prediction using our trained model `m`

````julia
ypred = predict_after_train(m,xtest)[1]
````

Now we can compare the true vs predicted results:

````julia
using Plots
gr()
plot(ytest',ypred)
````
