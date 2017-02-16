import Plots: scatter!, plot
import Reactive: Signal, value
import Interact: checkbox
import StatsBase: sample
#using Knet
import Knet: sigm
import StatsBase: WeightVec
export iniWeights, train_net, predict_after_train, loadSeasonal, RNNModel
#import Interact: checkbox
### Predict function using the ragged Array directly (deprecated)
include("helper_macros.jl")

"""
    abstract FluxModel

A supertype for all models to be defined within this package
"""
abstract FluxModel


## Should move to stable code, when performance ok
function mseLoss(w, x, y, model)
    p = predict(model,w,x)
    n = mapreduce(length,+,y)
    return mapreduce(ii->sumabs2(iix-iiy for (iix,iiy) in zip(ii...)),+,zip(p,y))/n
end

## Should move to stable code, when performance ok
function mseLoss_old(w, x, y, model)
    return sumabs2(predict_old(model,w, x) .- y) / size(y,2)
end

derivActivation(y,dy) = y.*(1-y).*dy # Maybe this should be matrix mult of dh * dh'
derivActivation!(dest,hidden,dh) = for j=1:length(hidden) dest[j]=hidden[j]*(1-hidden[j])*dh[j] end
deriv(::typeof(mseLoss),ytrue,ypred)=ytrue-ypred
deriv(::typeof(mseLoss_old),ytrue,ypred)=ytrue-ypred
function sigm(xi::Number)
  if xi>=0
    z=exp(-xi)
    return one(xi)/(one(xi)+z)
  else
    z=exp(xi)
    return z/(one(xi)+z)
  end
end

### Random number from a range [min, max) -- Fabian, ok like this???
import Distributions: Uniform
function rand_flt(min::Number, max::Number, dims...)
    return rand(Uniform(min,max),dims...)
end

### for 1/sqrt(n) rule of thumb cf. http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/

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
    ### How often will the losses of the whole dataset be evaluated
    losscalcsize=20,
    ### Also graphical output via the react/interact interface?
    plotProgress=false,
	### How many to plot in MOD vs OBS scatter plot
	nPlotsample=2000
    )

    nSamp = length(x)
    nVar  = size(x,1)

    lossesTrain = model.lossesTrain
    lossesVali = model.lossesVali
    outputtimesteps = model.outputtimesteps

    ## Define the used loss-Function based on the method to predict and the function defining how the predictions
    ## are compared with the data (default is Mean Squared Error across sequences)
    loss(w, x, y)=lossFunc(w, x, y, model)

    ### Normalize y to 0,1
    xNorm,xMin,xMax = normalize_data(x,-1.0,1.0)
    yNorm,yMin,yMax = normalize_data(y,0.0,1.0)
    model.xNorm,model.yNorm,model.yMin,model.yMax,model.xMin,model.xMax = xNorm,yNorm,yMin[1],yMax[1],xMin,xMax
    ## Split data set for cross-validation
    ## Could be parameterized in the function call later!!
    ## Also missing a full k-fold crossvalidation
    trainIdx, valiIdx = split_data(nSamp, batchSize)

    xTrain, xVali     = xNorm[trainIdx],xNorm[valiIdx]
    yTrain, yVali     = yNorm[trainIdx],yNorm[valiIdx]

    w = model.weights

    info("Starting training....")
    info( length(trainIdx), " Training samples, ",  length(valiIdx), " Validation samples. " )
    info("BatchSize: " , batchSize)

    ### Loss before training
    push!(lossesTrain,loss(w, xNorm[trainIdx], yNorm[trainIdx]))
    push!(lossesVali,loss(w, xNorm[valiIdx], yNorm[valiIdx]))
    push!(outputtimesteps,isempty(outputtimesteps) ? 1 : outputtimesteps[end]+1)
    curPredAll=predict(model, w, xNorm)
    info("Before training loss, Training set: ", lossesTrain[end], " Validation: ", lossesVali[end])


    ### Just for plotting performace selec not too many points (otherwise my notebook freezes etc)
    if plotProgress

      plotSampleTrain = sampleRagged(yTrain,nPlotsample)
      plotSampleVali  = sampleRagged(yVali,nPlotsample)

      predTrain       = curPredAll[trainIdx]
      predVali        = curPredAll[valiIdx]

      p=plotSignal(outputtimesteps,lossesTrain, lossesVali,
                                   extractRaggedSample(predTrain,plotSampleTrain), extractRaggedSample(yTrain,plotSampleTrain),
                                   extractRaggedSample(predVali, plotSampleVali) , extractRaggedSample(yVali, plotSampleVali))
      plot(p)
    end


    for i=1:nEpoch

        ### Batch approach: cylce randomly through subset of samples (==> more stochasticitym better performance (in theory))
        ### I don't understand why the performance does not get much better with smaller batches
        nTrSamp = length(trainIdx)
        batchIdx= sample(1:nTrSamp,batchSize, replace=false)
        ### Calc the loss gradient dloss/dw based on the current weight vector and the sample
        ### This is done here with the predef Adagrad method. Could be done explicitely to speed up
        dw = predict_with_gradient(model,model.weights,xTrain[batchIdx], yTrain[batchIdx],lossFunc)
        ### Update w according to loss gradient and algorithm (incl, parameters therein)
        w, params = update!(w, dw, searchParams)

        ### Loss on training set and on validation set
        ### Early stopping based on the validation set could be implemented (when validation loss gets worse again)
        ### but it will be heuristic, because one has to smooth the loss series (with batch there is noise)
        if rem(i,losscalcsize) == 1
          push!(outputtimesteps,outputtimesteps[end]+losscalcsize)
          push!(lossesTrain,loss(w, xTrain, yTrain))
          push!(lossesVali, loss(w, xVali, yVali))
        end

        ### Output
        if rem(i, infoStepSize) == 1
            println("Epoch $i, Training: ", lossesTrain[end], " Validation: ", lossesVali[end])
            ## For graphical real time monitoring (see cell above)
            #println(typeof(yNorm))
            if plotProgress
              latestStart = outputtimesteps[end] - minimum([trunc(Int,outputtimesteps[end]*0.66) 1000])
              subTS       = findfirst(i->i>=latestStart,outputtimesteps):length(outputtimesteps)
              curPredAll  = predict(model,w, xNorm)
              predTrain   = curPredAll[trainIdx]
              predVali    = curPredAll[valiIdx]
              p[1]        = (outputtimesteps,lossesTrain)
              p[2]        = (extractRaggedSample(predVali, plotSampleVali) , extractRaggedSample(yVali, plotSampleVali))
              p[3]        = (outputtimesteps[subTS],lossesTrain[subTS])
              p[4]        = (outputtimesteps,lossesVali)
              p[6]        = (outputtimesteps[subTS],lossesVali[subTS])
              p[7]        = (outputtimesteps,lossesTrain)
              p[8]        = (extractRaggedSample(predTrain,plotSampleTrain), extractRaggedSample(yTrain,plotSampleTrain))
              plot(p)
            end
        end
    end
    return model
end

""" Given the trained model this function does predictions on another data set
it also does the back-transform of the predicted values from (0,1) to original range
whole thing should be improved. An object of Type TrainedModel should be used or sth like a list in R
"""
function predict_after_train(model::FluxModel, x)

    #istaskdone(model.trainTask) || error("Training not finished yet")
    xNorm = normalize_data(x)
    yNorm = predict_old(model,model.weights, xNorm)
    yPred = yNorm .* (model.yMax-model.yMin) .+ model.yMin

    return yPred
end

function normalize_data(x,newmin,newmax)
  xall=hcat(x...)
  xMin,xMax=zeros(size(xall,1)),zeros(size(xall,1))
  for i=1:size(xall,1)
    xv = xall[i,:]
    xMin[i],xMax[i]=minimum(xv[!isnan(xv)]),maximum(xv[!isnan(xv)])
  end
  xNorm=deepcopy(x)
  for (xx,xxNorm) in zip(x,xNorm)
    for j in 1:size(xx,2), v in 1:size(xx,1)
      xxNorm[v,j] = (newmax-newmin).* ((xx[v,j]-xMin[v])/(xMax[v]-xMin[v]))+newmin
    end
  end
  xNorm,xMin,xMax
end

function sampleRagged{T}(x::Vector{Matrix{T}},nsample)
    nel = map(i->Float64(length(i)-sum(isnan,i)),x)
    sum(nel) < nsample && (nsample=round(Int,sum(nel)))
    isamp = sample(collect(1:length(x)),WeightVec(nel),nsample,replace=true)
    its   = map(i->rand(1:(length(x[i])-sum(isnan,x[i]))),isamp)
    goodvals = map(i->find(map(j->!isnan(j),i)),x)
    its2  = copy(its)
    for i=1:length(its)
      its2[i] = goodvals[isamp[i]][its[i]]
    end
    return [(isamp[i],its2[i]) for i=1:length(its)]
end

@noinline function extractRaggedSample(ragged_a,samp)
  [ragged_a[i][j] for (i,j) in samp]
end

function split_data(nSamp,batchSize)
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
  return trainIdx, valiIdx
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
