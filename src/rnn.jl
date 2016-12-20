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
  outputtimesteps::Vector{Int}
  lossesTrain::Vector{Float64}
  lossesVali::Vector{Float64}
  yMin::Float64
  yMax::Float64
  yNorm::Array{Float64,3}
  xNorm::Array{Float64,3}
  xEx::Vector{Tuple{Float64,Float64}}
end

### (The same) predict function using a linear Vector of w (needed for update!)
### and extracting the relevant Matrices, this is faster (cf. Fabian tests and
### nd Adam does not easily work with the ragged vector)

function predict(::RNNModel,w, x) ### This implements w as a vector

  # Reshape weight vectors and create temp arrays
  nTimes,nSamp,nVar,nHid,ypred,w1,w2,w3,w4,w5,hidden_pre,xw1,hidprew2,xcur,hidden = RNN_init_pred(w,x)

  for s=1:nSamp  ## could this loop be parallelized into say 100 parallel instances? Even a speed up factor 10 would make quite some diff
    # Run the prediction
    RNN_predict_loop(s,nTimes,nVar,nHid,hidden,hidden_pre,hidprew2,ypred,xcur,x,xw1,w1,w2,w3,w4,w5)
  end

  return ypred
end

function predict_old(::RNNModel,w, x) ### This implements w as a vector
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

function RNN_init_pred(w,x)

  nTimes, nSamp, nVar = size(x)
  nHid = Int(-0.5*(nVar + 2) + sqrt(0.25*(nVar+2)^2-1+length(w)))

  ypred=Array{typeof(w[1])}(nTimes, nSamp)
  w1=reshape(w[1:nVar*nHid], nVar, nHid)
  w2=reshape(w[nVar*nHid+1:nVar*nHid+nHid*nHid], nHid, nHid)
  w3=reshape(w[nVar*nHid+nHid*nHid+1:nVar*nHid+nHid*nHid+nHid], nHid, 1)
  w4=reshape(w[nVar*nHid+nHid*nHid+nHid+1:nVar*nHid+nHid*nHid+nHid+nHid], 1, nHid)
  w5=reshape(w[nVar*nHid+nHid*nHid+nHid+nHid+1:nVar*nHid+nHid*nHid+nHid+nHid+1], 1)

  # Allocate temprary arrays
  hidden_pre = zeros(1,nHid)
  xw1        = zeros(1,nHid)
  hidprew2   = zeros(1,nHid)
  xcur       = zeros(1,nVar)
  hidden     = zeros(1, nHid,nTimes+1)

  return nTimes,nSamp,nVar,nHid,ypred,w1,w2,w3,w4,w5,hidden_pre,xw1,hidprew2,xcur,hidden
end

function RNN_predict_loop(s,nTimes,nVar,nHid,hidden,hidden_pre,hidprew2,ypred,xcur,x,xw1,w1,w2,w3,w4,w5)

    #Emtpy temporary arrays
    fill!(hidden,zero(eltype(hidden)))
    fill!(hidden_pre,zero(eltype(hidden_pre)))

    for i=1:nTimes
      ## w[1] weights from input to hidden
      ## w[2] weights hidden to hidden
      ## w[3] weights hidden to output
      ## w[4] bias to hidden
      ## w[5] bias to output

      # First copy current x variables to xcur
      for j=1:nVar xcur[j]=x[i,s,j] end

      #Then calculate x * w1 and hidden[:,:,i-1]*w2
      A_mul_B!(xw1,xcur,w1)
      A_mul_B!(hidprew2,hidden_pre,w2)

      #Add both together and apply activation function
      @inbounds for j=1:nHid
        hidden[1,j,i]=sigm(xw1[j]+hidprew2[j]+w4[j])
      end

      #Store hidden state in hidden_pre matrix
      copy!(hidden_pre,1,hidden,nHid*(i-1)+1,nHid)

      #Calculate prediction through hidden*w3+w5
      ypred[i,s] = sigm(dot(hidden_pre,w3) + w5[1])
    end

end

## Should move to stable code, when performance ok
function mseLoss(w, x, y, model)
    return sumabs2(predict(model,w, x) .- y) / size(y,2)
end

## Should move to stable code, when performance ok
function mseLoss_old(w, x, y, model)
    return sumabs2(predict_old(model,w, x) .- y) / size(y,2)
end

derivActivation(y,dy) = y.*(1-y).*dy # Maybe this should be matrix mult of dh * dh'
derivActivation!(dest,hidden,dh) = for j=1:length(hidden) dest[j]=hidden[j]*(1-hidden[j])*dh[j] end
deriv(::typeof(mseLoss),ytrue,ypred)=ytrue-ypred

function Knet.sigm(xi::Number)
  if xi>=0
    z=exp(-xi)
    return one(xi)/(one(xi)+z)
  else
    z=exp(xi)
    return z/(one(xi)+z)
  end
end


function predict_with_gradient(::RNNModel,w, x,ytrue,lossFunc) ### This implements w as a vector

  # Reshape weight vectors and create temp arrays
  nTimes,nSamp,nVar,nHid,ypred,w1,w2,w3,w4,w5,hidden_pre,xw1,hidprew2,xcur,hidden = RNN_init_pred(w,x)

  # Allocate additional arrays for derivative calculation
  dWxh, dWhh, dWhy = [zeros(size(w1)) for i=1:nSamp], [zeros(size(w2)) for i=1:nSamp], [zeros(size(w3)) for i=1:nSamp]
  dbh, dby         = [zeros(size(w4)) for i=1:nSamp], [zeros(size(w5)) for i=1:nSamp]
  dhnext           = zeros(Float64,1,nHid)
  dh               = zeros(nHid,1)
  dhraw            = zeros(nHid,1)

  for s=1:nSamp  ## could this loop be parallelized into say 100 parallel instances? Even a speed up factor 10 would make quite some diff

    # Run the prediction
    RNN_predict_loop(s,nTimes,nVar,nHid,hidden,hidden_pre,hidprew2,ypred,xcur,x,xw1,w1,w2,w3,w4,w5)

    # Run the derivative backward pass
    for i=nTimes:-1:1

      # Derivative of the loss function
      dy = deriv(lossFunc,ytrue[i,s],ypred[i,s])

      # Derivative of the activations function (still a scalar)
      dy2 = derivActivation(ypred[i,s],dy)

      #Copy current hidden state to temp array
      copy!(hidden_pre,1,hidden,nHid*(i-1)+1,nHid)

      # Get dWhy and dh in a single loop
      dWhycur=dWhy[s]
      @inbounds for j=1:nHid
        dWhycur[j]+=hidden_pre[j]*dy2
        dh[j]=w3[j]*dy2 + dhnext[j]
      end

      # Derivative of output-bias is trivial
      dby[s][1] += dy2

      # Now apply derivative of activation function to the hidden state
      # Be careful - derivative must be expressed as f'(f(x))
      derivActivation!(dhraw,hidden_pre,dh)

      # Get hidden-bias derivatives
      dbhcur=dbh[s]
      @inbounds for j=1:nHid dbhcur[j]  += dhraw[j] end

      # Copy current x state to temp array
      @inbounds for j=1:nVar xcur[j]=x[i,s,j] end

      # Get input-to-hidden derivatives
      gemm!('T','T',1.0,xcur,dhraw,1.0,dWxh[s])

      # Fill the hidden-pre array with the previous time step or zeros if we are at time step 1
      if i==1
        fill!(hidden_pre,0.0)
      else
        copy!(hidden_pre,1,hidden,nHid*(i-2)+1,nHid)
      end

      # Now get the hidden-to-hidden derivatives (dWhh[s]=hidden_pre' * dhraw')
      gemm!('T','T',1.0,hidden_pre,dhraw,1.0,dWhh[s])

      # And finakky save dh for the next (previous time step)
      At_mul_B!(dhnext,dhraw,w2)
    end

  end

  return -[reshape(sum(dWxh), nVar*nHid ); reshape(sum(dWhh), nHid*nHid) ; reshape(sum(dWhy),nHid) ; reshape(sum(dbh),nHid) ; reshape(sum(dby),1)]
end


#function loss(w, x, y)
#    return sumabs2(predict_from_wVect(w, x) .- y) / size(y,2)
#end


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




function RNNModel(nVar,nHid,w=iniWeights(nVar,nHid,"RNN"))

  totalNweights=nVar*nHid + nHid * nHid + 2*nHid + 1

  length(w) == totalNweights ||  error("Length of weights $(size(weights,1)) does not match needed length $totalNweights!")
  RNNModel(w,nHid,0,identity,Int[],Float64[],Float64[],NaN,NaN,zeros(0,0,0),zeros(0,0,0),Tuple{Float64,Float64}[])
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
    ### How often will the losses of the whole dataset be evaluated
    losscalcsize=20,
    ### Also graphical output via the react/interact interface?
    plotProgress=false,
	### How many to plot in MOD vs OBS scatter plot
	nPlotsample=2000
    )

    nTimes, nSamp, nVarX=size(x)

    lossesTrain = model.lossesTrain
    lossesVali = model.lossesVali
    outputtimesteps = model.outputtimesteps

    ## Define the used loss-Function based on the method to predict and the function defining how the predictions
    ## are compared with the data (default is Mean Squared Error across sequences)
    loss(w, x, y)=lossFunc(w, x, y, model)

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
    push!(lossesTrain,loss(w, xNorm[:, trainIdx,:], yNorm[:, trainIdx, :]))
    push!(lossesVali,loss(w, xNorm[:, valiIdx,:], yNorm[:, valiIdx, :]))
    push!(outputtimesteps,isempty(outputtimesteps) ? 1 : outputtimesteps[end]+1)
    curPredAll=predict(model,w, xNorm)
    info("Before training loss, Training set: ", lossesTrain[end], " Validation: ", lossesVali[end])

    irsig=Signal(false)

    ### Just for plotting performace selec not too many points (otherwise my notebook freezes etc)
    if plotProgress

      cb=checkbox(label="Interrupt",signal=irsig)
      display(cb)

      plotSampleTrain = sample(1:length(yNorm[:, trainIdx, :]), min(nPlotsample,length(yNorm[:, trainIdx, :])) , replace=false)
      plotSampleVali = sample(1:length(yNorm[:, valiIdx, :]), min(nPlotsample,length(yNorm[:, valiIdx, :])) , replace=false)

      p=plotSignal(outputtimesteps,lossesTrain, lossesVali,vec(curPredAll[:, trainIdx,:]), vec(yNorm[:,trainIdx,:]),vec(curPredAll[:, valiIdx,:]), vec(yNorm[:,valiIdx,:]))
      plot(p)
    end


    for i=1:nEpoch

        value(irsig) && break
        ### Batch approach: cylce randomly through subset of samples (==> more stochasticitym better performance (in theory))
        ### I don't understand why the performance does not get much better with smaller batches
        nTrSamp = length(trainIdx)
        batchIdx=sample(1:nTrSamp,batchSize, replace=false)
        ### Calc the loss gradient dloss/dw based on the current weight vector and the sample
        ### This is done here with the predef Adagrad method. Could be done explicitely to speed up
        #dw = lossgradient(w, xNorm[:,trainIdx[batchIdx] ,:], yNorm[:, trainIdx[batchIdx],:])
        dw = predict_with_gradient(model,model.weights,xNorm[:,trainIdx[batchIdx] ,:], yNorm[:, trainIdx[batchIdx],:],lossFunc)
        ### Update w according to loss gradient and algorithm (incl, parameters therein)
        w, params = update!(w, dw, searchParams)

        ### Loss on training set and on validation set
        ### Early stopping based on the validation set could be implemented (when validation loss gets worse again)
        ### but it will be heuristic, because one has to smooth the loss series (with batch there is noise)
        if rem(i,losscalcsize) == 1
          push!(outputtimesteps,outputtimesteps[end]+losscalcsize)
          push!(lossesTrain,loss(w, xNorm[:, trainIdx,:], yNorm[:, trainIdx, :]))
          push!(lossesVali,loss(w, xNorm[:, valiIdx,:], yNorm[:, valiIdx, :]))
        end

        ### Output
        if rem(i, infoStepSize) == 1
            println("Epoch $i, Training: ", lossesTrain[end], " Validation: ", lossesVali[end])
            ## For graphical real time monitoring (see cell above)
            #println(typeof(yNorm))
            if plotProgress
              latestStart=outputtimesteps[end] - minimum([trunc(Int,outputtimesteps[end]*0.66) 1000])
              subTS=findfirst(i->i>=latestStart,outputtimesteps):length(outputtimesteps)
              curPredAll=predict(model,w, xNorm)
              p[1]=(outputtimesteps,lossesTrain)
              p[2]=(vec(curPredAll[:,valiIdx,:])[plotSampleVali],vec(yNorm[:,valiIdx,:])[plotSampleVali])
              p[3]=(outputtimesteps[subTS],lossesTrain[subTS])
              p[4]=(outputtimesteps,lossesVali)
              p[6]=(outputtimesteps[subTS],lossesVali[subTS])
              p[7]=(outputtimesteps,lossesTrain)
              p[8]=(vec(curPredAll[:,trainIdx,:])[plotSampleTrain],vec(yNorm[:,trainIdx,:])[plotSampleTrain])
              plot(p)
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
