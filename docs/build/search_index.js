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
    "text": "Pages = [\n    \"quickstart.md\",\n    \"training.md\",\n    \"models.md\",\n]\nDepth = 1"
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
    "page": "A quick test",
    "title": "A quick test",
    "category": "page",
    "text": ""
},

{
    "location": "quickstart.html#A-quick-test-1",
    "page": "A quick test",
    "title": "A quick test",
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
    "location": "models.html#RNNFluxes.LSTMModel",
    "page": "Available Models",
    "title": "RNNFluxes.LSTMModel",
    "category": "Type",
    "text": "type LSTM\n\nImplementation of an LSTM.\n\n\n\n"
},

{
    "location": "models.html#RNNFluxes.RNNModel",
    "page": "Available Models",
    "title": "RNNFluxes.RNNModel",
    "category": "Type",
    "text": "type RNN\n\nImplementation of an RNN.\n\n\n\n"
},

{
    "location": "models.html#RNNFluxes.GenericModel",
    "page": "Available Models",
    "title": "RNNFluxes.GenericModel",
    "category": "Type",
    "text": "type GenericModel\n\nType wrapping a generic model that is differentiated with AutoGrad. Only a predict function is needed.\n\n\n\n"
},

{
    "location": "models.html#Available-Models-1",
    "page": "Available Models",
    "title": "Available Models",
    "category": "section",
    "text": "RNNFluxes.LSTMModelRNNFluxes.RNNModelRNNFluxes.GenericModel"
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

]}
