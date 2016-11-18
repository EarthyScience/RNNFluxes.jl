module RNN
import Plots: scatter!, plot
import Reactive: Signal
import Interact: checkbox
import ..Update: Adam, update!
import ..PlotProgress: plotSignal, plotSummary
import StatsBase: sample
using Knet
export iniWeights, train_net, predict_after_train
#import Interact: checkbox
### Predict function using the ragged Array directly (deprecated)
function predict(w, x)
    nTimes, nSamp, nVar = size(x)
    #nSamp = size(x, 2)
    ypred=Array{typeof(w[1][1])}(nTimes, nSamp) ## Currently only one y - Variable can eb predicted

    for s=1:nSamp
        hidden = zeros(eltype(w[1]),1, size(w[2],1)) ## w[2] connects the hidden nodes
        for i=1:nTimes
            ## w[1] weights from input to hidden (must have dim  nVar, nHid)
            ## w[2] weights hidden to hidden (must have dim  nHid, nHid)
            ## w[3] weights hidden to output
            ## w[4] bias to hidden
            ## w[5] bias to output

             hidden = sigm(x[i:i, s, :] * w[1] + w[4] + hidden * w[2])
            ypred[i,s]=sigm(hidden * w[3])[1] + w[5][1]

        end
    end
    return ypred
end

### (The same) predict function using a linear Vector of w (needed for update!)
### and extracting the relevant Matrices, this is faster (cf. Fabian tests and
### nd Adam does not easily work with the ragged vector)

function RNNpredict(w, x) ### This implements w as a vector

    nTimes, nSamp, nVar = size(x)
    nHid = Int(-0.5*(nVar + 2) + sqrt(0.25*(nVar+2)^2-1+length(w)))

    ypred=Array{typeof(w[1])}(nTimes, nSamp)
    w1=reshape(w[1:nVar*nHid], nVar, nHid)
    w2=reshape(w[nVar*nHid+1:nVar*nHid+nHid*nHid], nHid, nHid)
    w3=reshape(w[nVar*nHid+nHid*nHid+1:nVar*nHid+nHid*nHid+nHid], nHid, 1)
    w4=reshape(w[nVar*nHid+nHid*nHid+nHid+1:nVar*nHid+nHid*nHid+nHid+nHid], 1, nHid)
    w5=reshape(w[nVar*nHid+nHid*nHid+nHid+nHid+1:nVar*nHid+nHid*nHid+nHid+nHid+1], 1)


    for s=1:nSamp  ## could this loop be parallelized into say 100 parallel instances? Even a speed up factor 10 would make quite some diff
        hidden = zeros(eltype(w[1]),1, nHid)

            for i=1:nTimes
                ## w[1] weights from input to hidden
                ## w[2] weights hidden to hidden
                ## w[3] weights hidden to output
                ## w[4] bias to hidden
                ## w[5] bias to output
                hidden = sigm(x[i:i, s, :] * w1 + w4 + hidden * w2)
               # ypred[i,s]=sigm(hidden * w3)[1] + w5[1]
				ypred[i,s]=sigm(hidden * w3 + w5)[1]

            end
    end
    return ypred
end


function loss(w, x, y)
    return sumabs2(predict_from_wVect(w, x) .- y) / size(y,2)
end


### Update the weights based on the current gradient dw, and the "normal" (not Nesterov) momentum approach
### STRANGE THAT the momentum update in !update gives different (slower!) convergence
function train_step_mom(w, dw, v; lr=.001, momentum=0.9)
        for i in 1:length(w)
            v[i] = momentum * v[i] - lr * dw[i]
            w[i] += v[i]
        end
    return w, v
end

### Update the weights based on the current gradient dw, and the rmsprop approach
### e.g. http://cs231n.github.io/neural-networks-3/
function train_step_rmsprop(w, dw, cache; lr=0.002, decay=0.9, eps=1e-6)
        for i in 1:length(w)
            cache[i] = decay .* cache[i] .+ (1-decay) .* dw[i].^2
            w[i] += -lr .* dw[i] ./ (cache[i].^0.5 .+ eps)
        end

    return w, cache
end

### for sqrt(n) rule of thumb cf. http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/

### Random number from a range [min, max) -- Fabian, ok like this???
import Distributions: Uniform
function rand_flt(min::Number, max::Number, dims...)
    return rand(Uniform(min,max),dims...)
end

### for 1/sqrt(n) rule of thumb cf. http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/

function iniWeights(nVarX::Int=3, nHid::Int=12, NNtype="RNN")
    if NNtype=="RNN"
              weights = -1.0 .+ 2.0 .*
        [rand_flt(-1./sqrt(nVarX), 1./sqrt(nVarX), nVarX, nHid),  ## Input to hidden
            rand_flt(1./-sqrt(nHid), 1./sqrt(nHid),  nHid,nHid),  ## HIdden to hidden
            rand_flt(1./-sqrt(nHid), 1./sqrt(nHid),  nHid, 1), ## Hidden to output
            0.0 * rand(Float64, 1, nHid),  ## Bias to hidden initialized to zero
            0.0 * rand(Float64, 1)] ## bias to output to zero
        weights=raggedToVector(weights)

    end
    return weights
end

## Should move to stable code, when performance ok
function mseLoss(w, x, y, predFunc)
    return sumabs2(predFunc(w, x) .- y) / size(y,2)
end

#### Top-level for fitting Timeseries with different ANN architechtures
#### Here Hyperparameters will be set etc.

#Training:
## Input x[nTimes, nSamples, nVar]
##       y[nTimes, nSamples]
##       nEpochs
function train_net(
    # Predictors, target and number of Epochs (y is still 1 variable only, can be changed to multioutput, but for us I don't see an urgent application)
    x, y, nEpoch=201;
    ## defines how a prediction is turned into a loss-Value (e.g. sum of squares, sum of abs, weighted squares etc..)
    lossFunc=mseLoss,
    ## defines the predict function (RNN, LSTM, or could be any model!!)
    predFunc=RNNpredict,
    ## number of hidden nodes
    nHid=12,
    ### the initial weights (e.g. weights from a previous training can be used to start of well already)
    ### Yet, what is still not nice: one has to re-enter the type of model. Maybe better to define a model type
    ### which contains all the info, including the hidden nodes etc.
    weights=iniWeights(size(x,3),nHid, NNtpye="RNN"),
    ## How many samples are used in each epoch
    batchSize=1,
    ## Define search algorithm including parameters
    searchParams = Adam(weights; lr=0.01, beta1=0.9, beta2=0.95, t=1, eps=1e-6, fstm=zeros(weights), scndm=zeros(weights)),
    ### How often will be intermediate output given
    infoStepSize=100,
    ### Also graphical output via the react/interact interface?
    plotProgress=false
    )

    nTimes, nSamp, nVarX=size(x)
    totalNweights=nVarX*nHid + nHid * nHid + 2*nHid + 1

    size(weights,1) == totalNweights ||  error("Length of weights $(size(weights,1)) does not match needed length $totalNweights!")


    lossesTrain = Array(Float64, nEpoch)
    lossesVali = Array(Float64, nEpoch)

    ## Define the used loss-Function based on the method to predict and the function defining how the predictions
    ## are compared with the data (default is Mean Squared Error across sequences)
    loss(w, x, y)=lossFunc(w, x, y, predFunc)
    lossgradient=grad(loss)

    ### Normalize y to 0,1
    yMin, yMax = extrema(y)
    yNorm=(y-yMin)/(yMax-yMin)
    ### Normalize x to -1, 1 per Variable (not necessarily the best way to do it....)
    xEx=vec(extrema(x, [1,2]))
    xNorm=copy(x)
    for v in 1:size(xEx,1);
        xNorm[:,:,v] = 2.0.* ((x[:,:,v]-xEx[v][1])/(xEx[v][2]-xEx[v][1])-0.5)
    end

    ## Split data set for cross-validation
    ## Could be parameterized in the function call later!!
    ## Also missing a full k-fold crossvalidation
    valiFrac=0.2
    permut=shuffle(1:nSamp)

    valiNSamp = trunc(Int, valiFrac*nSamp)

    if valiNSamp >= 2
        valiIdx=permut[1:valiNSamp]
        trainIdx=permut[valiNSamp+1:end]
    else
        valiIdx=1:nSamp
        trainIdx=1:nSamp
        warn("Too few samples (only $valiNSamp) for creating validation set!! Full data set (n=$nSamp sequences) used for training...")
    end
    if length(trainIdx) < batchSize
        warn("Too few samples for batchSize $(batchSize)! BatchSize adjusted to full training size $(length(trainIdx))")
        batchSize=length(trainIdx)

    end


    #valiIdx=1:nSamp; trainIdx=1:nSamp
    #display(plot(layer(x = vec(xNorm)[1:1000], y = vec(x)[1:1000], Geom.point)))


    w = weights

    info("Starting training....")
    info( length(trainIdx), " Training samples, ",  length(valiIdx), " Validation samples. " )
    info("BatchSize: " , batchSize)

    ### Loss before training
    lossesPreTrain=loss(w, xNorm[:, trainIdx,:], yNorm[:, trainIdx, :])
    lossesPreVali=loss(w, xNorm[:, valiIdx,:], yNorm[:, valiIdx, :])
    println("Before training loss, Training set: ", lossesPreTrain, " Validation: ", lossesPreVali)

    ### Just for plotting performace select not too many points (otherwise my notebook freezes etc)
    if plotProgress
      plotSample = sample(1:length(yNorm[:, trainIdx, :]), min(1000,length(yNorm[:, trainIdx, :])) , replace=false)

      yvals=Signal((rand(200), rand(200), rand(1000).*10, rand(1000)./10))

      display(map(plotSignal, yvals))
    end

    ### Here would be could if one could stop the loop based on user input to a checkbox (does not work)
    ### (Fabian knows what mean)
   # b=checkbox()
   # display(b)
   # quitLoop=Signal(b.value)

    for i=1:nEpoch;
        ### Batch approach: cylce randomly through subset of samples (==> more stochasticitym better performance (in theory))
        ### I don't understand why the performance does not get much better with smaller batches
        nTrSamp = length(trainIdx)
        batchIdx=sample(1:nTrSamp,batchSize, replace=false)
        ### Calc the loss gradient dloss/dw based on the current weight vector and the sample
        ### This is done here with the predef Adagrad method. Could be done explicitely to speed up
        dw = lossgradient(w, xNorm[:,trainIdx[batchIdx] ,:], yNorm[:, trainIdx[batchIdx],:])

        ### Update w according to loss gradient and algorithm (incl, parameters therein)
        w, params = update!(w, dw, searchParams)

        ### Loss on training set and on validation set
        ### Early stopping based on the validation set could be implemented (when validation loss gets worse again)
        ### but it will be heuristic, because one has to smooth the loss series (with batch there is noise)
        lossesTrain[i]=loss(w, xNorm[:, trainIdx,:], yNorm[:, trainIdx, :])
        lossesVali[i]=loss(w, xNorm[:, valiIdx,:], yNorm[:, valiIdx, :])

        ### Output
        if rem(i, infoStepSize) == 1
            println("Epoch $i, Training: ", lossesTrain[i], " Validation: ", lossesVali[i])
            ## For graphical real time monitoring (see cell above)
            curPredAll=predFunc(w, xNorm)
            #println(typeof(yNorm))
            if plotProgress
              newData=(lossesTrain[1:max(200,i)], lossesVali[1:max(200,i)],vec(curPredAll[:, trainIdx,:])[plotSample], vec(yNorm[:,trainIdx,:])[plotSample])
              plotProgress && push!(yvals, newData)
            end
        end

    ### Here would be could if one could stop the loop based on user input to a checkbox (does not work)
    ### (Fabian knows what mean)
        # push!(quitLoop, b.value)
        #println(quitLoop, b)
        #if quitLoop.value == true; break; end
    end
    plotSummary(lossesTrain, lossesVali,w,xNorm,yNorm,predFunc(w,xNorm))
    sleep(2)
    return (predFunc, w, lossesTrain, lossesVali, (yMin, yMax), xEx)
end

### Given the trained model this function does predictions on another data set
### it also does the back-transform of the predicted values from (0,1) to original range
### whole thing should be improved. An object of Type TrainedModel should be used or sth like a list in R
function predict_after_train(trainResultTuple, x)

    xNorm=copy(x)
    xEx=trainResultTuple[6]
    yMin, yMax = trainResultTuple[5]
    predFunc=trainResultTuple[1]
    w=trainResultTuple[2]

    for v in 1:size(xEx,1);
       xNorm[:,:,v] = 2.0.* ((x[:,:,v]-xEx[v][1])/(xEx[v][2]-xEx[v][1])-0.5)
    end
     yNorm = predFunc(w, xNorm)
    yPred = yNorm .* (yMax-yMin) .+ yMin

    return yPred
end

## after http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
function LSTMpredict(w, x)

    ### So far only implemented with the nicely interpretable ragged w - Vector.
    ### TODO: accept w as linear vector and convert into ragged Vector (like in predict_from_wVEct in "../stableCode/NeuralNetUtils.jl")

    nTimes, nSamp, nVar = size(x)
    #nSamp = size(x, 2)
    ypred=Array{typeof(w[1][1])}(nTimes, nSamp) ## Currently only one y - Variable can eb predicted


    # wUi = w[1] # from x to input gate (nVar, nHid)
    # wWi = w[2] # from hidden to input gate (nHid, nhid)
   # wUf = w[3] # from x to forget gate
   # wWf = w[4] # from hidden to forget gate
   # wUo = w[5] # from x to output gate
   # wWo = w[6] # from hidden to output gate
   # wUg = w[7] # from x to candidate hidden update
   # wWg = w[8] # from hidden to candiate hidden update
   # Wout =  w[9] # from hidden to predictions
    biasInp = w[10] ### bias from input (1, nHid)
   biasOut =w[11] ### bias to output (1) (one output only)


    for s=1:nSamp
        hidden = zeros(eltype(w[1]),1, size(w[2],1)) ## w[2] connects the hidden nodes
        c=copy(hidden)
        for i=1:nTimes
            ## w[1] weights from input to hidden (must have dim  nVar, nHid)
            ## w[2] weights hidden to hidden (must have dim  nHid, nHid)
            ## w[3] weights hidden to output
            ## w[4] bias to hidden
            ## w[5] bias to output

            iGate = sigm(x[i:i, s, :] * w[1] + hidden * w[2] + biasInp)
            fGate = sigm(x[i:i, s, :] * w[3] + hidden * w[4] + biasInp)
            oGate = sigm(x[i:i, s, :] * w[5] + hidden * w[6] + biasInp)
            g = sigm(x[i:i, s, :] * w[7] + hidden * w[8] + biasInp)
            c = c .* fGate + g .* iGate
            hidden = tanh(c) .* oGate

            ypred[i,s]=sigm(hidden * w[9]+ biasOut)[1]

        end
    end
    return ypred
end

function raggedToVector(raggedArray)
    totalLength=0
    for elem in raggedArray totalLength += length(elem); end

    ragged_as_Vector=Array{typeof(raggedArray[1][1]), 1}(totalLength)
    totalLength=0

    for elem in raggedArray
        elemLength=length(elem)
        ragged_as_Vector[totalLength+1:totalLength+elemLength]= reshape(elem, elemLength)
        totalLength+=elemLength
    end
    return ragged_as_Vector
end

end