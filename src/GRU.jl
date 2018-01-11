import Base.LinAlg: gemv!
using Distributions

"""
    type GRU

Implementation of an GRU.
"""
type GRUModel <: FluxModel
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

function GRUModel(nVar,nHid; dist = Uniform)
  w=iniWeights(GRUModel, nVar, nHid, dist)
  totalNweights=3*(nVar * nHid + nHid * nHid + nHid) + nHid + 1
  length(w) == totalNweights ||  error("Length of weights $(size(weights,1)) does not match needed length $totalNweights!")
  GRUModel(w,nHid,0,identity,Int[],Float64[],Float64[],NaN,NaN,[zeros(0,0)],[zeros(0,0)],Float64[],Float64[])
end

function iniWeights(::Type{GRUModel}, nVarX::Int, nHid::Int, dist)
  weights = [rand_flt(1./sqrt(nVarX), dist, nHid, nVarX),  ## Update gate Z
  rand_flt(1./sqrt(nHid), dist, nHid, nHid),
  rand_flt(1./sqrt(nHid), dist, nHid, 1),
  rand_flt(1./sqrt(nVarX), dist, nHid, nVarX),  ## reset gate R
  rand_flt(1./sqrt(nHid),  dist, nHid,nHid),
  rand_flt(1./sqrt(nHid),  dist, nHid, 1),
  rand_flt(1./sqrt(nVarX), dist, nHid, nVarX),  ## Candidate Activation Gate h_tilde
  rand_flt(1./sqrt(nHid),  dist, nHid,nHid),
  rand_flt(1./sqrt(nHid),  dist, nHid, 1),
  rand_flt(1./sqrt(nHid), dist, 1, nHid), ## Linear activation weights
  rand_flt(1./sqrt(1), dist, 1, 1)] ## Linear activation bias
  weights = raggedToVector(weights)
end

function predict(model::GRUModel,w,x)
  nSamp = length(x)
  nVar  = size(x[1],1)

  nHid = model.nHid
  @reshape_weights(w1=>(nHid,nVar), w2=>(nHid,nHid), w3=>(nHid,1),
  w4=>(nHid,nVar), w5=>(nHid,nHid), w6=>(nHid,1),
  w7=>(nHid,nVar), w8=>(nHid,nHid), w9=>(nHid,1),
  w10=>(1,nHid),   w11=>(1,1))

  # This loop was rewritten using map so that one can easily switch to pmap later
  ypred = map(x) do xx

    nTimes = size(xx,2)
    yout   = zeros(nTimes)
    hidden = zeros(eltype(w[1]), nHid, 1)
    state  = copy(hidden)
    for i=1:nTimes
      zgate   = sigm(w1 * xx[:,i:i] + w2 * hidden + w3)
      rgate   = sigm(w4 * xx[:,i:i] + w5 * hidden + w6)
      state   = tanh(w7 * xx[:,i:i] + w8 * (hidden .* rgate) + w9)
      hidden   = hidden .* zgate + (1 .- zgate) .* state
      yout[i] = sigm(w10 * hidden + w11)[1]
    end
    yout
  end
  ypred
end

#gemv!(tA, alpha, A, x, beta, y)
#Update the vector y as alpha*A*x + beta*y or alpha*A'x + beta*y according to tA (transpose A). Returns the updated y.
# macro chain_matmulv_add(ex)
#   ex.head==:(.=) || error("Wrong format. Input expression must be assignment with .=")
#   outAr = ex.args[1]
#   ex = ex.args[2]
#   (ex.head==:call && ex.args[1]==:(+)) || error("Wrong input format, right-hand side must be a sum")
#   outEx = quote end
#   for a in ex.args[2:end]
#     if isa(a,Symbol)
#       push!(outEx.args,:(vec_add!($outAr,$a)))
#     elseif a.head==:call && a.args[1]==:(*)
#       matsym = a.args[2]
#       t='N'
#       if isa(matsym,Expr)
#         t='T'
#         matsym=matsym.args[1]
#       end
#       vecsym = a.args[3]
#       push!(outEx.args,:(gemv!($t,1.0,$matsym,$vecsym,1.0,$outAr)))
#     else
#       error("Unknown operand")
#     end
#   end
#   outEx
# end


# "Adds vectors a and b and stores the result in a"
# function vec_add!(a,b)
#   length(a) == length(b) || error("Lengths of a and b differ")
#   @inbounds for i=1:length(a)
#     a[i]=a[i]+b[i]
#   end
#   a
# end

function predict_with_gradient(model::GRUModel,w, x,ytrue,lossFunc) ### This implements w as a vector
  nTimes = size(x[1],2)
  nSamp = length(x)
  nVar = size(x[1],1)
  nHid = model.nHid

  @reshape_weights(w1=>(nHid,nVar),  w2=>(nHid,nHid), w3=>(nHid,1),    w4=>(nHid,nVar), w5=>(nHid,nHid),
  w6=>(nHid,1),  w7=>(nHid,nVar), w8=>(nHid,nHid), w9=>(nHid,1), w10=>(1,nHid), w11=>(1,1))

  # Allocate additional arrays for derivative calculation
  ypred=Array{typeof(w1[1])}(nTimes, nSamp)
  dw1, dw4, dw7 = [zeros(size(w1)) for i=1:nSamp], [zeros(size(w4)) for i=1:nSamp], [zeros(size(w7)) for i=1:nSamp]
  dw2, dw5, dw8 = [zeros(size(w2)) for i=1:nSamp], [zeros(size(w5)) for i=1:nSamp], [zeros(size(w8)) for i=1:nSamp]
  dw3, dw6, dw9 = [zeros(size(w3)) for i=1:nSamp], [zeros(size(w6)) for i=1:nSamp], [zeros(size(w9)) for i=1:nSamp]
  dw10, dw11 = [zeros(size(w10)) for i=1:nSamp], [zeros(size(w11)) for i=1:nSamp]

  for s=1:nSamp
    hidden = zeros(nTimes + 1, nHid)
    zgate, rgate, state = zeros(nTimes+1, nHid), zeros(nTimes+1, nHid), Array{Float64}(nTimes, nHid)
    dHidden = zeros(nHid,1)
    dZGate, dRGate, dState = zeros(nTimes+1,nHid), zeros(nTimes+1,nHid), zeros(nTimes+1,nHid)

    for i=1:nTimes
      zgate[i,:]   = sigm(w1 * x[s][:,i:i] + w2 * hidden[i ,:] + w3)
      rgate[i,:]   = sigm(w4 * x[s][:,i:i] + w5 * hidden[i ,:] + w6)
      state[i,:]   = tanh(w7 * x[s][:,i:i] + w8 * (hidden[i ,:] .* rgate[i ,:]) + w9)
      hidden[i+1,:]= hidden[i ,:] .* zgate[i,:] + (1 .- zgate[i ,:]) .* state[i ,:]
      ypred[i,s] = sigm(w10 * hidden[i+1,:] + w11)[1]
    end

    # Run the derivative backward pass
    for i=nTimes:-1:1

      # Derivative of the loss function
      dy = deriv(lossFunc, ytrue[s][i], ypred[i,s])
      # Derivative of the activations function (still a scalar)
      dy2 = derivActivation(ypred[i,s], dy)
      dw10[s] += dy2 * hidden[i+1,:]'
      dw11[s][1] += dy2

      dHidden = w10' * dy2 + rgate[i+1,:] .* (w8' * dState[i+1,:]) + w5' * dRGate[i+1,:] + w2' * dZGate[i+1,:] + dHidden .* zgate[i+1,:]
      dState[i,:] = dHidden .* (1 .- zgate[i,:]) .* (1 .- (state[i,:] .* state[i,:]))
      dRGate[i,:] = hidden[i,:] .* (w8' * dState[i,:]) .* rgate[i,:] .* (1 .- rgate[i,:])
      dZGate[i,:] = dHidden .* (hidden[i,:] .- state[i,:]) .* zgate[i,:] .* (1 .- zgate[i,:])

      # Update input weights
      dw1[s] += dZGate[i, :] * x[s][:,i:i]'#x[i:i, s, :]
      dw4[s] += dRGate[i, :] * x[s][:,i:i]'#x[i:i, s, :]
      dw7[s] += dState[i, :] * x[s][:,i:i]'#x[i:i, s, :]
      # Update hidden weights
      dw2[s] += dZGate[i+1,:] *  hidden[i+1,:]'
      dw5[s] += dRGate[i+1,:] *  (hidden[i+1,:] .* rgate[i+1,:])'
      dw8[s] += dState[i+1,:] *  hidden[i+1,:]'
      # Update biases
      dw3[s] += dZGate[i+1,:]
      dw6[s] += dRGate[i+1,:]
      dw9[s] += dState[i+1,:]
    end
  end
  return -[reshape(sum(dw1), nVar*nHid); reshape(sum(dw2), nHid*nHid); reshape(sum(dw3), nHid);
  reshape(sum(dw4), nVar*nHid); reshape(sum(dw5), nHid*nHid); reshape(sum(dw6), nHid);
  reshape(sum(dw7), nVar*nHid); reshape(sum(dw8), nHid*nHid); reshape(sum(dw9), nHid);
  reshape(sum(dw10), nHid); reshape(sum(dw11), 1)]
end
