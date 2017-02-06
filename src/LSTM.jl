"""
    type LSTM

Implementation of an LSTM.
"""
type LSTMModel <: FluxModel
  weights::Vector{Float64}
  nHid::Int
  nVarX::Int
  lossFunc::Function
  outputtimesteps::Vector{Int}
  lossesTrain::Vector{Float64}
  lossesVali::Vector{Float64}
  yMin::Float64
  yMax::Float64
  yNorm::Vector{Array{Float64,2}}
  xNorm::Vector{Array{Float64,2}}
  xMin::Vector{Float64}
  xMax::Vector{Float64}
end

function LSTMModel(nVar,nHid)
  w=iniWeights(LSTMModel,nVar,nHid)
  totalNweights=4*(nVar * nHid + nHid * nHid + nHid) + nHid + 1
  length(w) == totalNweights ||  error("Length of weights $(size(weights,1)) does not match needed length $totalNweights!")
  LSTMModel(w,nHid,0,identity,Int[],Float64[],Float64[],NaN,NaN,[zeros(0,0)],[zeros(0,0)],Float64[],Float64[])
end

function predict(model::LSTMModel,w,x)
  nSamp = length(x)
  nVar  = size(x[1],1)

  nHid = model.nHid
  dim1 = Int(nVar * nHid + nHid * nHid + nHid)
  @reshape_weights(w1=>(nHid,nVar),  w2=>(nHid,nHid), w3=>(nHid,1),    w4=>(nHid,nVar), w5=>(nHid,nHid),
                   w6=>(nHid,1),     w7=>(nHid,nVar), w8=>(nHid,nHid), w9=>(nHid,1),    w10=>(nHid,nVar),
                   w11=>(nHid,nHid), w12=>(nHid,1),   w13=>(1,nHid),   w14=>(1,1))

  # This loop was rewritten using map so that one can easily switch to pmap later
  ypred = map(x) do xx

    nTimes = size(xx,2)
    yout   = zeros(nTimes)
    hidden = zeros(eltype(w[1]),1, nHid)
    state  = copy(hidden')
    for i=1:nTimes
        a       = tanh(w1 * xx[:,i:i] + w2 * hidden' + w3)
        igate   = sigm(w4 * xx[:,i:i] + w5 * hidden' + w6)
        fgate   = sigm(w7 * xx[:,i:i] + w8 * hidden' + w9)
        ogate   = sigm(w10 *xx[:,i:i] + w11 * hidden' + w12)
        state   = a .* igate + fgate .* state
        hidden  = (tanh(state) .* ogate)'
        yout[i] = sigm(w13 * hidden' + w14)[1]
    end
    yout
  end
  ypred
end

function predict_with_gradient(model::LSTMModel,w, x,ytrue,lossFunc) ### This implements w as a vector
  nSamp = length(x)

  nVar = size(x[1],1)
  nHid = model.nHid

  @reshape_weights(w1=>(nHid,nVar),  w2=>(nHid,nHid), w3=>(nHid,1),    w4=>(nHid,nVar), w5=>(nHid,nHid),
                   w6=>(nHid,1),     w7=>(nHid,nVar), w8=>(nHid,nHid), w9=>(nHid,1),    w10=>(nHid,nVar),
                   w11=>(nHid,nHid), w12=>(nHid,1),   w13=>(1,nHid),   w14=>(1,1))

  # Allocate additional arrays for derivative calculation
  dw1, dw4, dw7, dw10 = [zeros(size(w1)) for i=1:nSamp], [zeros(size(w4)) for i=1:nSamp], [zeros(size(w7)) for i=1:nSamp], [zeros(size(w10)) for i=1:nSamp]
  dw2, dw5, dw8, dw11 = [zeros(size(w2)) for i=1:nSamp], [zeros(size(w5)) for i=1:nSamp], [zeros(size(w8)) for i=1:nSamp], [zeros(size(w11)) for i=1:nSamp]
  dw3, dw6, dw9, dw12 = [zeros(size(w3)) for i=1:nSamp], [zeros(size(w6)) for i=1:nSamp], [zeros(size(w9)) for i=1:nSamp], [zeros(size(w12)) for i=1:nSamp]
  dw13, dw14 = [zeros(size(w13)) for i=1:nSamp], [zeros(size(w14)) for i=1:nSamp]

  foreach(x,ytrue,1:nSamp) do xx,yy,s
    nTimes = size(xx,2)
    ypred  = zeros(nTimes)
    out, state = zeros(nTimes + 1, nHid), zeros(nTimes + 1, nHid)
    dState, dOut  = zeros(nHid,1), zeros(nHid,1)
    dInput, dIgate, dFgate, dOgate = [zeros(nHid) for i=1:nTimes+1], [zeros(nHid) for i=1:nTimes+1], [zeros(nHid) for i=1:nTimes+1], [zeros(nHid) for i=1:nTimes+1]
    input, igate, fgate, ogate = Array{Float64}(nTimes, nHid),Array{Float64}(nTimes, nHid),zeros(nTimes+1, nHid),Array{Float64}(nTimes, nHid)

    for i=1:nTimes
        xslice       = xx[:,i:i]
        outslice     = out[i,:]
        input[i,:]   = tanh(w1 * xslice + w2 * outslice + w3)
        igate[i,:]   = sigm(w4 * xslice + w5 * outslice + w6)
        fgate[i,:]   = sigm(w7 * xslice + w8 * outslice + w9)
        ogate[i ,:]  = sigm(w10 * xslice + w11 * outslice + w12)
        state[i+1,:] = input[i ,:] .* igate[i ,:] + fgate[i ,:] .* state[i ,:]
        out[i+1,:]   = tanh(state[i+1 ,:]) .* ogate[i ,:]
        ypred[i]     = sigm(w13 * out[i+1,:] + w14)[1]
    end

    dInput1, dIgate1,dFgate1,dOgate1 = dInput[nTimes+1],dIgate[nTimes+1],dFgate[nTimes+1],dOgate[nTimes+1]
    # Run the derivative backward pass
    for i=nTimes:-1:1

      # Derivative of the loss function
      dy = deriv(lossFunc,yy[i],ypred[i])
      # Derivative of the activations function (still a scalar)
      dy2 = derivActivation(ypred[i],dy)
      dw13[s] += dy2 * out[i+1,:]'
      dw14[s][1] += dy2

      dOut = w13' * dy2 + w2' * dInput1 + w5' * dIgate1 + w8' * dFgate1 + w11' * dOgate1 ## Most important line

      # Update hidden weights
      if i<nTimes
        outslice = out[i+1,:]
        gemm!('N','T',1.0,dInput1,outslice,1.0,dw2[s])
        gemm!('N','T',1.0,dIgate1,outslice,1.0,dw5[s])
        gemm!('N','T',1.0,dFgate1,outslice,1.0,dw8[s])
        gemm!('N','T',1.0,dOgate1,outslice,1.0,dw11[s])
      end
      # Update biases
      for j=1:nHid
        dw3[s][j] += dInput1[j]
        dw6[s][j] += dIgate1[j]
        dw9[s][j] += dFgate1[j]
        dw12[s][j] += dOgate1[j]
      end

      dInput1, dIgate1,dFgate1,dOgate1 = dInput[i], dIgate[i], dFgate[i], dOgate[i]

      dState = dOut .* ogate[i,:] .* (1 - tanh(state[i+1,:]) .* tanh(state[i+1,:])) + dState .* fgate[i+1,:]     ## Also very important
      for j=1:nHid
        dInput1[j] = dState[j] * igate[i,j] * (1 - input[i,j] * input[i,j])
        dIgate1[j] = dState[j] * input[i,j] * igate[i,j] * (1 - igate[i,j])
        dFgate1[j] = dState[j] * state[i,j] * fgate[i,j] * (1 - fgate[i,j])
        dOgate1[j] = dOut[j] * tanh(state[i+1,j]) * ogate[i,j] * (1 - ogate[i,j])
      end
      # Update input weights
      xslice = xx[:,i:i]
      gemm!('N','T',1.0,dInput1,xslice,1.0,dw1[s])
      gemm!('N','T',1.0,dIgate1,xslice,1.0,dw4[s])
      gemm!('N','T',1.0,dFgate1,xslice,1.0,dw7[s])
      gemm!('N','T',1.0,dOgate1,xslice,1.0,dw10[s])
    end
  end

  return -[reshape(sum(dw1), nVar*nHid); reshape(sum(dw2), nHid*nHid); reshape(sum(dw3), nHid);
  reshape(sum(dw4), nVar*nHid); reshape(sum(dw5), nHid*nHid); reshape(sum(dw6), nHid);
  reshape(sum(dw7), nVar*nHid); reshape(sum(dw8), nHid*nHid); reshape(sum(dw9), nHid);
  reshape(sum(dw10), nVar*nHid); reshape(sum(dw11), nHid*nHid); reshape(sum(dw12), nHid);
  reshape(sum(dw13), nHid); reshape(sum(dw14), 1)]
  #return -[reshape(sum(dWxh), nVar*nHid ); reshape(sum(dWhh), nHid*nHid) ; reshape(sum(dWhy),nHid) ; reshape(sum(dbh),nHid) ; reshape(sum(dby),1)]
end

function iniWeights(::Type{LSTMModel},nVarX::Int=3, nHid::Int=12)
  weights = -1.0 + 2.0 .*
  [rand_flt(-1./sqrt(nVarX), 1./sqrt(nVarX), nHid, nVarX),  ## Input block
  rand_flt(1./-sqrt(nHid), 1./sqrt(nHid),  nHid,nHid),
  rand_flt(1./-sqrt(nHid), 1./sqrt(nHid),  nHid, 1),
  rand_flt(-1./sqrt(nVarX), 1./sqrt(nVarX), nHid, nVarX),  ## Input gate
  rand_flt(1./-sqrt(nHid), 1./sqrt(nHid),  nHid,nHid),
  rand_flt(1./-sqrt(nHid), 1./sqrt(nHid),  nHid, 1),
  rand_flt(-1./sqrt(nVarX), 1./sqrt(nVarX), nHid, nVarX),  ## Forget Gate
  rand_flt(1./-sqrt(nHid), 1./sqrt(nHid),  nHid,nHid),
  rand_flt(1./-sqrt(nHid), 1./sqrt(nHid),  nHid, 1),
  rand_flt(-1./sqrt(nVarX), 1./sqrt(nVarX), nHid, nVarX),  ## Output Gate
  rand_flt(1./-sqrt(nHid), 1./sqrt(nHid),  nHid,nHid),
  rand_flt(1./-sqrt(nHid), 1./sqrt(nHid),  nHid, 1),
  0.0 * rand(Float64, 1, nHid), ## Linear activation weights
  0.0 * rand(Float64, 1)] ## Linear activation bias
  weights=raggedToVector(weights)
end
