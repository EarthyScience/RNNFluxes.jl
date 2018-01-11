import Base.LinAlg.BLAS: gemv!, gemm!
using Distributions

"""
    type LSTMModel

Implementation of an LSTM with handcoded gradients. It has the following constructor:

    LSTMModel(nVar,nHid; dist = Uniform, forgetBias = 1, nDropout=nHid ÷ 10)

### Parameters

* `nVar` number of input (predictor) variables for each time step
* `nHid` number of hidden nodes contained in the LSTM
* `dist` distribution to generate initial Weights, currently `Uniform` and `Normal` are supported, defaults to `Uniform`
* `forgetBias` determine bias of generating forget bias weights, defaults to 1
* `nDropOut` number of dropout nodes for each prediction, defaults to 10% of the nodes
"""
type LSTMModel <: FluxModel
  weights::Vector{Float64}
  nHid::Int
  nVarX::Int
  lossFunc::Function
  outputtimesteps::Vector{Int}
  lossesTrain::Vector{Float64}
  lossesVali::Vector{Float64}
  yMin::Vector{Float64}
  yMax::Vector{Float64}
  yNorm::Vector{Array{Float64,2}}
  xNorm::Vector{Array{Float64,2}}
  xMin::Vector{Float64}
  xMax::Vector{Float64}
  n_dropout::Int64
end

function LSTMModel(nVar,nHid; dist = Uniform, forgetBias = 1, nDropout=nHid ÷ 10)
  w=iniWeights(LSTMModel, nVar, nHid, dist, forgetBias)
  totalNweights=4*(nVar * nHid + nHid * nHid + nHid) + nHid + 1
  length(w) == totalNweights ||  error("Length of weights $(size(weights,1)) does not match needed length $totalNweights!")
  LSTMModel(w,nHid,0,identity,Int[],Float64[],Float64[],Float64[],Float64[],[zeros(0,0)],[zeros(0,0)],Float64[],Float64[],nDropout)
end

function iniWeights(::Type{LSTMModel}, nVarX::Int, nHid::Int, dist, forgetBias)
  weights = [rand_flt(1./sqrt(nVarX), dist, nHid, nVarX),  ## Input block
  rand_flt(1./sqrt(nHid), dist, nHid, nHid),
  rand_flt(1./sqrt(nHid), dist, nHid, 1),
  rand_flt(1./sqrt(nVarX), dist, nHid, nVarX),  ## Input gate
  rand_flt(1./sqrt(nHid),  dist, nHid,nHid),
  rand_flt(1./sqrt(nHid),  dist, nHid, 1),
  rand_flt(1./sqrt(nVarX), dist, nHid, nVarX),  ## Forget Gate
  rand_flt(1./sqrt(nHid),  dist, nHid,nHid),
  forgetBias != 0 ? forgetBias .* ones(nHid, 1) : rand_flt(1./sqrt(nHid),  dist, nHid, 1),
  rand_flt(1./sqrt(nVarX), dist, nHid, nVarX),  ## Output Gate
  rand_flt(1./sqrt(nHid),  dist, nHid,nHid),
  rand_flt(1./sqrt(nHid),  dist, nHid, 1),
  rand_flt(1./sqrt(nHid), dist, 1, nHid), ## Linear activation weights
  rand_flt(1./sqrt(1), dist, 1, 1)] ## Linear activation bias
  weights = raggedToVector(weights)
end

function predict(model::LSTMModel,w,x)#;record_hidden::Bool=false, hidAr::Vector{Matrix{Float64}}=Matrix{Float64}[])
  nSamp = length(x)
  nVar  = size(x[1],1)

  nHid = model.nHid
  @reshape_weights(w1=>(nHid,nVar),  w2=>(nHid,nHid), w3=>(nHid,1),    w4=>(nHid,nVar), w5=>(nHid,nHid),
  w6=>(nHid,1),     w7=>(nHid,nVar), w8=>(nHid,nHid), w9=>(nHid,1),    w10=>(nHid,nVar),
  w11=>(nHid,nHid), w12=>(nHid,1),   w13=>(1,nHid),   w14=>(1,1))

  # This loop was rewritten using map so that one can easily switch to pmap later
  ypred = Vector{Float64}[]



  for xx in x

    nTimes = size(xx,2)
    yout   = zeros(nTimes)
    hidden = zeros(eltype(w[1]),nHid, 1)
    out,state   = zeros(nHid),zeros(nHid),zeros(nHid),zeros(nHid)
    state  = zeros(nHid)
    input, igate, fgate, ogate = zeros(nHid),zeros(nHid),zeros(nHid),zeros(nHid)
    xHelp         = zeros(nHid)
    state  = copy(hidden)
    for i=1:nTimes
      xslice      = xx[:,i]
      @chain_matmulv_add(xHelp.=w1  * xslice + w2  * out + w3 ); map!(tanh,input,xHelp); fill!(xHelp,0.0)
      @chain_matmulv_add(xHelp.=w4  * xslice + w5  * out + w6 ); map!(sigm,igate,xHelp); fill!(xHelp,0.0)
      @chain_matmulv_add(xHelp.=w7  * xslice + w8  * out + w9 ); map!(sigm,fgate,xHelp); fill!(xHelp,0.0)
      @chain_matmulv_add(xHelp.=w10 * xslice + w11 * out + w12); map!(sigm,ogate,xHelp); fill!(xHelp,0.0)
      @inbounds for j=1:nHid
        out[j]   = tanh(state[j]) * ogate[j]
        state[j] = input[j] * igate[j] + fgate[j] * state[j]
      end
      yout[i]    = sigm(dot(w13,out) + w14[1])
    end
    push!(ypred,yout)
  end
  ypred
end



function predict_with_gradient{T}(model::LSTMModel,w::AbstractVector{T}, x,ytrue,lossFunc) ### This implements w as a vector
  nSamp = length(x)

  nVar = size(x[1],1)
  nHid = model.nHid

  @reshape_weights(w1=>(nHid,nVar),  w2=>(nHid,nHid), w3=>(nHid,1),    w4=>(nHid,nVar), w5=>(nHid,nHid),
  w6=>(nHid,1),     w7=>(nHid,nVar), w8=>(nHid,nHid), w9=>(nHid,1),    w10=>(nHid,nVar),
  w11=>(nHid,nHid), w12=>(nHid,1),   w13=>(1,nHid),   w14=>(1,1))

  # println(macroexpand(:(@reshape_weights(w1=>(nHid,nVar),  w2=>(nHid,nHid), w3=>(nHid,1),    w4=>(nHid,nVar), w5=>(nHid,nHid),
  #                  w6=>(nHid,1),     w7=>(nHid,nVar), w8=>(nHid,nHid), w9=>(nHid,1),    w10=>(nHid,nVar),
  #                  w11=>(nHid,nHid), w12=>(nHid,1),   w13=>(1,nHid),   w14=>(1,1)))))

  #println(isdefined(:w1))
  #println(isdefined(:w2))

  idropout = sample(1:nHid,model.n_dropout,replace=false)
  w2[idropout,:]=0.0
  w5[idropout,:]=0.0
  w8[idropout,:]=0.0
  w11[idropout,:]=0.0
  w13[:,idropout]=0.0

  # Allocate additional arrays for derivative calculation
  @allocate_dw w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14

  foreach(x,ytrue,1:nSamp) do xx,yy,s
    nTimes = size(xx,2)
    ypred  = zeros(nTimes)
    out, state = [zeros(nHid) for i = 1:nTimes+1], [zeros(nHid) for i=1:nTimes+1]
    dState, dOut  = zeros(nHid,1), zeros(nHid)
    xHelp         = zeros(nHid)
    dInput, dIgate, dFgate, dOgate = [zeros(nHid) for i=1:nTimes+1], [zeros(nHid) for i=1:nTimes+1], [zeros(nHid) for i=1:nTimes+1], [zeros(nHid) for i=1:nTimes+1]
    input, igate, fgate, ogate = [zeros(nHid) for i=1:nTimes],[zeros(nHid) for i=1:nTimes],[zeros(nHid) for i=1:nTimes+1],[zeros(nHid) for i=1:nTimes]

    for i=1:nTimes
      xslice      = xx[:,i]
      @chain_matmulv_add(xHelp.=w1  * xslice + w2  * out[i] + w3 ); map!(tanh,input[i],xHelp); fill!(xHelp,0.0)
      @chain_matmulv_add(xHelp.=w4  * xslice + w5  * out[i] + w6 ); map!(sigm,igate[i],xHelp); fill!(xHelp,0.0)
      @chain_matmulv_add(xHelp.=w7  * xslice + w8  * out[i] + w9 ); map!(sigm,fgate[i],xHelp); fill!(xHelp,0.0)
      @chain_matmulv_add(xHelp.=w10 * xslice + w11 * out[i] + w12); map!(sigm,ogate[i],xHelp); fill!(xHelp,0.0)
      input1,igate1,fgate1,ogate1,state1,out2,state2 = input[i],igate[i],fgate[i],ogate[i],state[i],out[i+1],state[i+1]
      @inbounds for j=1:nHid
        state2[j] = input1[j] * igate1[j] + fgate1[j] * state1[j]
        out2[j]   = tanh(state2[j]) * ogate1[j]
      end
      ypred[i]    = sigm(dot(w13,out[i+1]) + w14[1])
    end

    dInput1, dIgate1,dFgate1,dOgate1 = dInput[nTimes+1],dIgate[nTimes+1],dFgate[nTimes+1],dOgate[nTimes+1]

    # Run the derivative backward pass
    for i=nTimes:-1:1

      # Derivative of the loss function
      dy = deriv(lossFunc,yy,ypred,i)
      # Derivative of the activations function (still a scalar)
      dy2 = derivActivation(ypred[i],dy)

      dw13_s = dw13[s]; out_i1 = out[i+1]
      @inbounds for j=1:nHid
        dw13_s[j] += dy2 * out_i1[j]
      end
      dw14[s][1] += dy2

      dOut[:]=0.0
      @chain_matmulv_add(dOut .= w13' * [dy2] + w2' * dInput1 + w5' * dIgate1 + w8' * dFgate1 + w11' * dOgate1) ## Most important line

      # Update hidden weights
      if i<nTimes
        outslice = out[i+1]
        gemm!('N','T',1.0,dInput1,outslice,1.0,dw2[s])
        gemm!('N','T',1.0,dIgate1,outslice,1.0,dw5[s])
        gemm!('N','T',1.0,dFgate1,outslice,1.0,dw8[s])
        gemm!('N','T',1.0,dOgate1,outslice,1.0,dw11[s])
      end
      # Update biases
      @inbounds for j=1:nHid
        dw3[s][j] += dInput1[j]
        dw6[s][j] += dIgate1[j]
        dw9[s][j] += dFgate1[j]
        dw12[s][j] += dOgate1[j]
      end

      dInput1, dIgate1,dFgate1,dOgate1 = dInput[i], dIgate[i], dFgate[i], dOgate[i]
      input1,   igate1, fgate1, ogate1 =  input[i],  igate[i],  fgate[i],  ogate[i]
      state2,fgate2 = state[i+1], fgate[i+1]

      #map!((dO,og,st,dS,fg)->dO * og * (1-tanh(st)) * tanh(st) + dS * fg,dState,dOut,ogate1,state[i+1],dState,fgate[i+1])
      @inbounds for j=1:nHid
        dState[j] = dOut[j] * ogate1[j] * (1 - tanh(state2[j]) * tanh(state2[j])) + dState[j] * fgate2[j]
      end     ## Also very important
      @inbounds for j=1:nHid
        dInput1[j] = dState[j] * igate1[j] * (1 - input1[j] * input1[j])
        dIgate1[j] = dState[j] * input1[j] * igate1[j] * (1 - igate1[j])
        dFgate1[j] = dState[j] * state[i][j] * fgate1[j] * (1 - fgate1[j])
        dOgate1[j] = dOut[j] * tanh(state[i+1][j]) * ogate1[j] * (1 - ogate1[j])
      end
      # Update input weights
      xslice = xx[:,i:i]
      gemm!('N','T',1.0,dInput1,xslice,1.0,dw1[s])
      gemm!('N','T',1.0,dIgate1,xslice,1.0,dw4[s])
      gemm!('N','T',1.0,dFgate1,xslice,1.0,dw7[s])
      gemm!('N','T',1.0,dOgate1,xslice,1.0,dw10[s])
    end
  end

  #Set derivatives of dropped out nodes to 0
  dw2_2 = sum(dw2)
  dw5_2 = sum(dw5)
  dw8_2 = sum(dw8)
  dw11_2= sum(dw11)
  dw13_2= sum(dw13)
  dw2_2[idropout,:]=0.0
  dw5_2[idropout,:]=0.0
  dw8_2[idropout,:]=0.0
  dw11_2[idropout,:]=0.0
  dw13_2[:,idropout]=0.0


  return -[reshape(sum(dw1), nVar*nHid); reshape(dw2_2, nHid*nHid) ; reshape(sum(dw3), nHid);
  reshape(sum(dw4), nVar*nHid); reshape(dw5_2, nHid*nHid); reshape(sum(dw6), nHid);
  reshape(sum(dw7), nVar*nHid); reshape(dw8_2, nHid*nHid); reshape(sum(dw9), nHid);
  reshape(sum(dw10), nVar*nHid); reshape(dw11_2, nHid*nHid); reshape(sum(dw12), nHid);
  reshape(dw13_2, nHid); reshape(sum(dw14), 1)]
end
