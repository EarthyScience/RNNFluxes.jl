import Base.LinAlg: gemv!

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

#gemv!(tA, alpha, A, x, beta, y)
#Update the vector y as alpha*A*x + beta*y or alpha*A'x + beta*y according to tA (transpose A). Returns the updated y.
macro chain_matmulv_add(ex)
  ex.head==:(.=) || error("Wrong format. Input expression must be assignment with .=")
  outAr = ex.args[1]
  ex = ex.args[2]
  (ex.head==:call && ex.args[1]==:(+)) || error("Wrong input format, right-hand side must be a sum")
  outEx = quote end
  for a in ex.args[2:end]
    if isa(a,Symbol)
      push!(outEx.args,:(vec_add!($outAr,$a)))
    elseif a.head==:call && a.args[1]==:(*)
      matsym = a.args[2]
      t='N'
      if isa(matsym,Expr)
        t='T'
        matsym=matsym.args[1]
      end
      vecsym = a.args[3]
      push!(outEx.args,:(gemv!($t,1.0,$matsym,$vecsym,1.0,$outAr)))
    else
      error("Unknown operand")
    end
  end
  outEx
end

macroexpand(:(@chain_matmulv_add dOut.=w1*x+w2*y+w3'*z+w4))

"Adds vectors a and b and stores the result in a"
function vec_add!(a,b)
  length(a) == length(b) || error("Lengths of a and b differ")
  @inbounds for i=1:length(a)
    a[i]=a[i]+b[i]
  end
  a
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
      dy = deriv(lossFunc,yy[i],ypred[i])
      # Derivative of the activations function (still a scalar)
      dy2 = [derivActivation(ypred[i],dy)]
      dw13[s] += dy2[1] * out[i+1]'
      dw14[s][1] += dy2[1]

      @chain_matmulv_add(dOut .= w13' * dy2 + w2' * dInput1 + w5' * dIgate1 + w8' * dFgate1 + w11' * dOgate1) ## Most important line

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
