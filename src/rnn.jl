import Plots: scatter!, plot
import Reactive: Signal, value
import Interact: checkbox
import StatsBase: sample
using Knet
export iniWeights, train_net, predict_after_train, loadSeasonal, RNNModel
#import Interact: checkbox
### Predict function using the ragged Array directly (deprecated)

"""
    abstract FluxModel

A supertype for all models to be defined within this package
"""
abstract FluxModel

"""
    type RNN

Implementation of an RNN.
"""
type RNNModel <: FluxModel
  weights::Vector{Float64}
  nHid::Int
  nVarX::Int
  lossFunc::Function
  trainTask::Task
  lossesTrain::Vector{Float64}
  lossesVali::Vector{Float64}
  yMin::Float64
  yMax::Float64
  yNorm::Array{Float64,3}
  xNorm::Array{Float64,3}
  xEx::Vector{Tuple{Float64,Float64}}
  interruptSignal::Signal{Bool}
end

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

function predict(::RNNModel,w, x) ### This implements w as a vector

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
				        ypred[i,s] = sigm(hidden * w3 + w5)[1]
            end
    end
    return ypred
end

derivActivation(dh) = dot(sigm(dh),(1-sigm(dh))) # Maybe this should be matrix mult of dh * dh'
derivloss(ytrue,ypred)=ytrue-ypred

function predict_with_gradient(::RNNModel,w, x,ytrue) ### This implements w as a vector

  nTimes, nSamp, nVar = size(x)
  nHid = Int(-0.5*(nVar + 2) + sqrt(0.25*(nVar+2)^2-1+length(w)))

  ypred=Array{typeof(w[1])}(nTimes, nSamp)
  w1=reshape(w[1:nVar*nHid], nVar, nHid)
  w2=reshape(w[nVar*nHid+1:nVar*nHid+nHid*nHid], nHid, nHid)
  w3=reshape(w[nVar*nHid+nHid*nHid+1:nVar*nHid+nHid*nHid+nHid], nHid, 1)
  w4=reshape(w[nVar*nHid+nHid*nHid+nHid+1:nVar*nHid+nHid*nHid+nHid+nHid], 1, nHid)
  w5=reshape(w[nVar*nHid+nHid*nHid+nHid+nHid+1:nVar*nHid+nHid*nHid+nHid+nHid+1], 1)

  dWxh, dWhh, dWhy = [zeros(size(w1)) for i=1:nSamp], [zeros(size(w2)) for i=1:nSamp], [zeros(size(w3)) for i=1:nSamp]
  dbh, dby = [zeros(size(w4)) for i=1:nSamp], [zeros(size(w5)) for i=1:nSamp]
  dhnext = 0

  for s=1:nSamp  ## could this loop be parallelized into say 100 parallel instances? Even a speed up factor 10 would make quite some diff

    hidden = zeros(eltype(w[1]),1, nHid,nTimes+1)
    dhnext = 0

    for i=1:nTimes
      ## w[1] weights from input to hidden
      ## w[2] weights hidden to hidden
      ## w[3] weights hidden to output
      ## w[4] bias to hidden
      ## w[5] bias to output
      hidden[:,:,i] = sigm(x[i:i, s, :] * w1 + w4 + hidden[:,:,max(i,1)] * w2)
      # ypred[i,s]=sigm(hidden * w3)[1] + w5[1]
      ypred[i,s]=sigm(hidden[:,:,i] * w3 + w5)[1]

    end


    for i=nTimes:-1:1
      ## w[1] weights from input to hidden
      ## w[2] weights hidden to hidden
      ## w[3] weights hidden to output
      ## w[4] bias to hidden
      ## w[5] bias to output
      dy=derivloss(ytrue[i,s],ypred[i,s])
      dWhy[s]+=dy*hidden[:,:,i]'
      dby[s]+=dy
      dh = w3 * dy + dhnext
      dhraw = derivActivation(hidden[:,:,i])*dh
      dbh[s]  += dhraw'
      dWxh[s] += (dhraw * x[i:i,s,:])'
      dWhh[s] += dhraw * (i==1 ? zeros(1,nHid) : hidden[:,:,i-1])
      dhnext = w2' * dhraw
      # ypred[i,s]=sigm(hidden * w3)[1] + w5[1
      if s==1
        println(i)
        println(dWxh[s])
      end
    end
  end

  return ypred, dWxh, dWhh, dWhy, dbh, dby
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
function mseLoss(w, x, y, model)
    return sumabs2(predict(model,w, x) .- y) / size(y,2)
end


function RNNModel(nVar,nHid,w=iniWeights(nVar,nHid,"RNN"))

  totalNweights=nVar*nHid + nHid * nHid + 2*nHid + 1

  length(w) == totalNweights ||  error("Length of weights $(size(weights,1)) does not match needed length $totalNweights!")
  RNNModel(w,nHid,0,identity,Task(identity),Float64[],Float64[],NaN,NaN,zeros(0,0,0),zeros(0,0,0),Tuple{Float64,Float64}[],Signal(false))
end

function train(model::RNNModel,x,y,nEpoch=201;kwargs...)
  model.trainTask=@schedule train_net(model,x,y,nEpoch;kwargs...)
  sleep(1)
end

#Training:
## Input x[nTimes, nSamples, nVar]
##       y[nTimes, nSamples]
##       nEpochs
function train_net(
    model::FluxModel,
    # Predictors, target and number of Epochs (y is still 1 variable only, can be changed to multioutput, but for us I don't see an urgent application)
    x, y, nEpoch=201;
    ## defines how a prediction is turned into a loss-Value (e.g. sum of squares, sum of abs, weighted squares etc..)
    lossFunc=mseLoss,
    ## How many samples are used in each epoch
    batchSize=1,
    ## Define search algorithm including parameters
    searchParams = Adam(model.weights; lr=0.01, beta1=0.9, beta2=0.95, t=1, eps=1e-6, fstm=zeros(model.weights), scndm=zeros(model.weights)),
    ### How often will be intermediate output given
    infoStepSize=100,
    ### Also graphical output via the react/interact interface?
    plotProgress=false
    )

    nTimes, nSamp, nVarX=size(x)

    lossesTrain = model.lossesTrain
    lossesVali = model.lossesVali

    ## Define the used loss-Function based on the method to predict and the function defining how the predictions
    ## are compared with the data (default is Mean Squared Error across sequences)
    loss(w, x, y)=lossFunc(w, x, y, model)
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
    model.yNorm, model.xNorm = yNorm, xNorm
    model.yMin, model.yMax, model.xEx = yMin, yMax, xEx

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


    w = model.weights

    info("Starting training....")
    info( length(trainIdx), " Training samples, ",  length(valiIdx), " Validation samples. " )
    info("BatchSize: " , batchSize)

    ### Loss before training
    lossesPreTrain=loss(w, xNorm[:, trainIdx,:], yNorm[:, trainIdx, :])
    lossesPreVali=loss(w, xNorm[:, valiIdx,:], yNorm[:, valiIdx, :])
    info("Before training loss, Training set: ", lossesPreTrain, " Validation: ", lossesPreVali)

    irsig=model.interruptSignal

    ### Just for plotting performace select not too many points (otherwise my notebook freezes etc)
    if plotProgress

      cb=checkbox(label="Interrupt",signal=irsig)
      display(cb)
      plotSample = sample(1:length(yNorm[:, trainIdx, :]), min(1000,length(yNorm[:, trainIdx, :])) , replace=false)
      yvals=Signal((rand(200), rand(200), rand(1000).*10, rand(1000)./10))
      display(map(plotSignal, yvals))

    end


    ### Here would be could if one could stop the loop based on user input to a checkbox (does not work)
    ### (Fabian knows what mean)
   # b=checkbox()
   # display(b)
   # quitLoop=Signal(b.value)

    for i=1:nEpoch

        value(irsig) && break
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
        push!(lossesTrain,loss(w, xNorm[:, trainIdx,:], yNorm[:, trainIdx, :]))
        push!(lossesVali,loss(w, xNorm[:, valiIdx,:], yNorm[:, valiIdx, :]))

        ### Output
        if rem(i, infoStepSize) == 1
            println("Epoch $i, Training: ", lossesTrain[i], " Validation: ", lossesVali[i])
            ## For graphical real time monitoring (see cell above)
            curPredAll=predict(model,w, xNorm)
            #println(typeof(yNorm))
            if plotProgress
              newData=(lossesTrain, lossesVali,vec(curPredAll[:, trainIdx,:])[plotSample], vec(yNorm[:,trainIdx,:])[plotSample])
              plotProgress && push!(yvals, newData)
            end
        end
    end
    return model
end

### Given the trained model this function does predictions on another data set
### it also does the back-transform of the predicted values from (0,1) to original range
### whole thing should be improved. An object of Type TrainedModel should be used or sth like a list in R
function predict_after_train(model::FluxModel, x)

    #istaskdone(model.trainTask) || error("Training not finished yet")
    xNorm=copy(x)

    for v in 1:size(model.xEx,1);
       xNorm[:,:,v] = 2.0.* ((x[:,:,v]-model.xEx[v][1])/(model.xEx[v][2]-model.xEx[v][1])-0.5)
    end
    yNorm = predict(model,model.weights, xNorm)
    yPred = yNorm .* (model.yMax-model.yMin) .+ model.yMin

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
