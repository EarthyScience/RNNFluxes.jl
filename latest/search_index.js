var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#RNNFluxes.jl-1",
    "page": "Home",
    "title": "RNNFluxes.jl",
    "category": "section",
    "text": "A specialized RNN and LSTM implementation hand-coded in Julia for CPU"
},

{
    "location": "index.html#Package-features-1",
    "page": "Home",
    "title": "Package features",
    "category": "section",
    "text": "Fast training and prediction of 1-layer LSTM and RNN models of small to medium complexity."
},

{
    "location": "index.html#Manual-Outline-1",
    "page": "Home",
    "title": "Manual Outline",
    "category": "section",
    "text": "Pages = [\n    \"quickstart.md\",\n    \"training.md\",\n    \"models.md\",\n    \"example_lossfunctions.md\"\n]\nDepth = 1"
},

{
    "location": "index.html#Acknowledgements-1",
    "page": "Home",
    "title": "Acknowledgements",
    "category": "section",
    "text": "This package was written by Markus Reichstein, Fabian Gans and Nikolai Huckle at the Max-Planck-Institute for Biogeochemistry, Jena, Germany."
},

{
    "location": "index.html#Index-1",
    "page": "Home",
    "title": "Index",
    "category": "section",
    "text": ""
},

{
    "location": "quickstart.html#",
    "page": "A quick Overview",
    "title": "A quick Overview",
    "category": "page",
    "text": ""
},

{
    "location": "quickstart.html#A-quick-Overview-1",
    "page": "A quick Overview",
    "title": "A quick Overview",
    "category": "section",
    "text": "Here is a quick demonstration of how to train and apply an LSTM. Lets create some artificial input data first:nTime=100\nnSample=200\nnVarX = 3\nx = [rand(nVarX,nTime) for i=1:nSample]\n\ntrue_model(x) = transpose(cumsum(x[1,:])+exp.(x[2,:])+cumprod(1.7*x[3,:]))\n\ny = true_model.(x);Let's see what we have here. x contains the predictor variables. It is contains 200 samples of multivariate time series. Each sample is a nVar x nTime matrix containing the inputs for the model. Our function true_model generates a new time series from such a multivariate time series and is applied to every input sample. So our target variable y contains nSample row Vectors of length ntime.Now we create a model:using RNNFluxes\nnHid=15\nm=RNNFluxes.LSTMModel(nVarX,nHid,nDropout=1)This initializes an LSTM model containing 15 hidden nodes which represent the persistent state of the model. To train the model we can run:train_net(m,x,y,5001);This trains the model m for 5001 update steps. Note that the model variable m is updated in-place. To test how well our model performs, we generate a test dataset:xtest = [rand(nVarX,nTime)]\nytest = true_model.(xtest)[1]and do a prediction using our trained model mypred = predict_after_train(m,xtest)[1]Now we can compare the true vs predicted results:using Plots\ngr()\nplot(ytest',ypred)"
},

{
    "location": "models.html#",
    "page": "Available Models",
    "title": "Available Models",
    "category": "page",
    "text": ""
},

{
    "location": "models.html#Available-Models-1",
    "page": "Available Models",
    "title": "Available Models",
    "category": "section",
    "text": ""
},

{
    "location": "models.html#RNNFluxes.LSTMModel",
    "page": "Available Models",
    "title": "RNNFluxes.LSTMModel",
    "category": "Type",
    "text": "type LSTMModel\n\nImplementation of an LSTM with handcoded gradients. It has the following constructor:\n\nLSTMModel(nVar,nHid; dist = Uniform, forgetBias = 1, nDropout=nHid ÷ 10)\n\nParameters\n\nnVar number of input (predictor) variables for each time step\nnHid number of hidden nodes contained in the LSTM\ndist distribution to generate initial Weights, currently Uniform and Normal are supported, defaults to Uniform\nforgetBias determine bias of generating forget bias weights, defaults to 1\nnDropOut number of dropout nodes for each prediction, defaults to 10% of the nodes\n\n\n\n"
},

{
    "location": "models.html#RNNFluxes.RNNModel",
    "page": "Available Models",
    "title": "RNNFluxes.RNNModel",
    "category": "Type",
    "text": "type RNN\n\nImplementation of an RNN with handcoded gradients.\n\nRNNModel(nVar,nHid;dist=Uniform)\n\nParameters\n\nnVar number of input (predictor) variables for each time step\nnHid number of hidden nodes contained in the RNN\ndist distribution to generate initial Weights, currently Uniform and Normal are supported, defaults to Uniform\n\n\n\n"
},

{
    "location": "models.html#Recurrent-models-with-hand-coded-gradients-1",
    "page": "Available Models",
    "title": "Recurrent models with hand-coded gradients",
    "category": "section",
    "text": "RNNFluxes.LSTMModelRNNFluxes.RNNModel"
},

{
    "location": "models.html#RNNFluxes.GenericModel",
    "page": "Available Models",
    "title": "RNNFluxes.GenericModel",
    "category": "Type",
    "text": "type GenericModel\n\nType wrapping a generic model that is differentiated with AutoGrad. Only a predict function is needed.\n\n\n\n"
},

{
    "location": "models.html#Trainning-a-generic-model-using-AutoDiff-1",
    "page": "Available Models",
    "title": "Trainning a generic model using AutoDiff",
    "category": "section",
    "text": "We have also wrapped the training and plotting machinery here together with the ForwardDiff package, so that you can invert any custom model, too. Please note that, for this prupose the package (Flux.jl)[https://github.com/FluxML/Flux.jl] might be a better and well-maintained option.  RNNFluxes.GenericModelA short example on how to define and train such a generic model follows. Let's assume we already know the model structure but want to estimate the parameters. Any Julia code would be valid here in defining this model:function my_generic_model{T}(model,w::AbstractArray{T},x)\n\n    a = w[1]\n    b = w[2]\n    c = w[3]\n\n    map(x) do ix\n        cumsum(a * ix[1,:])+exp.(b * ix[2,:])+cumprod(c * ix[3,:])\n    end\nendWe generate some artificial data again:nTime=100\nnSample=200\nnVarX = 3\nx = [rand(nVarX,nTime) for i=1:nSample]\n\ntrue_model(x) = transpose(cumsum(x[1,:])+exp.(x[2,:])+cumprod(1.7*x[3,:]))\n\ny = true_model.(x);In the next step we define a GenericModel which includes our above-mentioned predict function.params = ()\nw      = rand(3) #Init weights\nm      = RNNFluxes.GenericModel(params,w,my_ANN_model)Training works \"as usual\"t = train_net(m,x,y,4001);"
},

{
    "location": "training.html#",
    "page": "Training and predicting models",
    "title": "Training and predicting models",
    "category": "page",
    "text": ""
},

{
    "location": "training.html#RNNFluxes.train_net",
    "page": "Training and predicting models",
    "title": "RNNFluxes.train_net",
    "category": "Function",
    "text": "train_net(model::FluxModel, x, y, nEpoch)\n\nTrains a model defined in model on the given predictors x and labels y for nEpoch training steps. Here, x as well as y are vectors of matrices of size (nVar x nTimeStep). Each matrix stands for a single sample, for example a tower time series. The length of the individual time series is allowed to differ.\n\nKeyword arguments\n\nlossFunc lossFUnction to be evaluated, defaults to mseLoss\nbatchSize number of samples to be used for each update step\nsearchParams One of the training methods defined in Knet. Can be any object of type Sgd, Momentum, Adam, Adagrad, Adadelta, Rmsprop. Have a look at update.jl to see how to construct this\ninfoStepSize stepsize in which traing and validation losses will be calculated and printed to the screen\nvaliFrac fraction of samples to be omitted from training as a validation dataset. Can be either a Float for a random fraction or vector of indices of omitted datasets, defaults to 0.2\nlosscalcsize determines how often the losses of the whole dataset will be evaluated so that the training process can be analyzed afterwards\nplotProgress shall the training be visualized, works only in Jupyter notebooks\nnPlotsample number of sample points used for plotting\n\n\n\n"
},

{
    "location": "training.html#RNNFluxes.predict_after_train",
    "page": "Training and predicting models",
    "title": "RNNFluxes.predict_after_train",
    "category": "Function",
    "text": "Given the trained model this function does predictions on another data set it also does the back-transform of the predicted values from (0,1) to original range whole thing should be improved. An object of Type TrainedModel should be used or sth like a list in R\n\n\n\n"
},

{
    "location": "training.html#Training-and-predicting-models-1",
    "page": "Training and predicting models",
    "title": "Training and predicting models",
    "category": "section",
    "text": "train_netpredict_after_train"
},

{
    "location": "example_lossfunctions.html#",
    "page": "Example loss functions",
    "title": "Example loss functions",
    "category": "page",
    "text": ""
},

{
    "location": "example_lossfunctions.html#Example-loss-functions-1",
    "page": "Example loss functions",
    "title": "Example loss functions",
    "category": "section",
    "text": "For training a model well, it is essential to provide a suitable loss function, which in many cases is more complicated than a simple MSE. To give some inspiration, here are a few examples of loss functions and derivatives that we have used in our work. The function signature of the loss function should be: (w::Vector, x::Vector{Matrix},y::Vector{Matrix},model::FluxModel)->Number, where w is the weight vector, x the predictors, y the true targets and model is the model applied. The function returns a single number. The derivative is defined by overloading the RNNFluxes.deriv function with a new method depending on the loss function type. The signature is (t::Type{Function},ytrue,ypred,i)->Number where t is the type of the loss function, ytrue is a time series of observations, ypred a predicted time series and i the index for which the derivative is calculated."
},

{
    "location": "example_lossfunctions.html#MSE-with-missing-values-1",
    "page": "Example loss functions",
    "title": "MSE with missing values",
    "category": "section",
    "text": "In case there are gaps in the target values, they can be omitted from MSE calculation. We assume here that gaps are encoded as NaNs.function mselossMiss(w, x, y, model)\n    p = RNNFluxes.predict(model,w,x)\n    n = mapreduce(i->sum(a->!isnan(a),i),+,y)\n    return mapreduce(ii->sumabs2(iix-iiy for (iix,iiy) in zip(ii...) if !isnan(iiy)),+,zip(p,y))/n\nend\nRNNFluxes.deriv(::typeof(mselossMiss),ytrue::Vector,ypred::Vector,i::Integer)=isnan(ytrue[i]) ? zero(ytrue[i]-ypred[i]) : ytrue[i]-ypred[i]"
},

{
    "location": "example_lossfunctions.html#Mixing-loss-for-each-single-time-step-and-difference-of-aggregated-fluxes-for-a-full-year-1",
    "page": "Example loss functions",
    "title": "Mixing loss for each single time step and difference of aggregated fluxes for a full year",
    "category": "section",
    "text": "As a more complex example, here we combine a normal MSE loss function with one that computes the loss only aggregated on annual values. The factor α determines the ratio to which the individual vs the annual loss are weighted.function mselossMissYear(w, x, y, model)\n    p = RNNFluxes.predict(model,w,x)\n    #Do the annual aggregation\n    lossesPerSite = map(y,p) do ytrue,ypred\n        nY = length(ytrue)÷NpY\n        lossTot = zero(eltype(ypred))\n        nTot    = 0\n        for iY = 1:nY\n            s,sp,n = annMeanN(ytrue,ypred,NpY,iY)\n            lossTot += (s - sp)^2\n            nTot += n\n        end\n        lossTot,nTot\n    end\n    lossAnnual = sum(i->i[1],lossesPerSite)/sum(i->i[2],lossesPerSite)\n    nSingle = mapreduce(i->sum(a->!isnan(a),i),+,y)\n    lossSingle = mapreduce(ii->sum(abs2(iix-iiy) for (iix,iiy) in zip(ii...) if !isnan(iiy)),+,zip(p,y))/nSingle\n    return α * lossAnnual + (1-α) * lossSingle\nend\n\n\n"
},

]}
