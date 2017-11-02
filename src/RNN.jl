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
  yNorm::Vector{Matrix{Float64}}
  xNorm::Vector{Matrix{Float64}}
  xMin::Vector{Float64}
  xMax::Vector{Float64}
end

function RNNModel(nVar,nHid;dist=Uniform)
  w=iniWeights(RNNModel,nVar,nHid,dist)
  totalNweights=nVar*nHid + nHid * nHid + 2*nHid + 1
  length(w) == totalNweights ||  error("Length of weights $(size(weights,1)) does not match needed length $totalNweights!")
  RNNModel(w,nHid,0,identity,Int[],Float64[],Float64[],NaN,NaN,[zeros(0,0)],[zeros(0,0)],Float64[],Float64[])
end

function iniWeights(::Type{RNNModel},nVarX::Int, nHid::Int, dist)
  weights = [rand_flt(1./sqrt(nVarX), dist, nHid, nVarX), ## Input to hidden
  rand_flt(1./sqrt(nHid), dist, nHid, nHid), # Hidden to hidden
  rand_flt(1./sqrt(nHid), dist, nHid, 1), # Hidden to Output
  rand_flt(1./sqrt(nHid), dist, 1, nHid), ## Linear activation weights
  rand_flt(1./sqrt(1), dist, 1, 1)] ## Linear activation bias
  weights=raggedToVector(weights)
end

function predict(m::RNNModel,w, x) ### This implements w as a vector

  # Reshape weight vectors and create temp arrays
  nSamp,nVar,nHid,w1,w2,w3,w4,w5,hidden_pre,xw1,hidprew2,xcur = RNN_init_pred(m,w,x)

  y=map(x) do xs  ## could this loop be parallelized into say 100 parallel instances? Even a speed up factor 10 would make quite some diff

    nTimes=size(xs,2)
    hidden = zeros(1, nHid,nTimes+1)
    ypred=zeros(nTimes)
    # Run the prediction
    RNN_predict_loop(nTimes,nVar,nHid,hidden,hidden_pre,hidprew2,ypred,xcur,xs,xw1,w1,w2,w3,w4,w5)
    ypred
  end

  return y
end

function RNN_init_pred(model::RNNModel,w,x)
  nSamp = length(x)
  nHid = model.nHid
  nVar = size(x[1],1)

  @reshape_weights(w1=>(nVar,nHid), w2=>(nHid,nHid), w3=>(nHid,1), w4=>(1,nHid), w5=>1)

  # Allocate temprary arrays
  hidden_pre = zeros(1,nHid)
  xw1        = zeros(1,nHid)
  hidprew2   = zeros(1,nHid)
  xcur       = zeros(1,nVar)
  return nSamp,nVar,nHid,w1,w2,w3,w4,w5,hidden_pre,xw1,hidprew2,xcur
end

function predict_with_gradient(m::RNNModel,w, x,ytrue,lossFunc) ### This implements w as a vector

  # Reshape weight vectors and create temp arrays
  nSamp,nVar,nHid,w1,w2,w3,w4,w5,hidden_pre,xw1,hidprew2,xcur = RNN_init_pred(m,w,x)

  # Allocate additional arrays for derivative calculation
  dWxh, dWhh, dWhy = [zeros(size(w1)) for i=1:nSamp], [zeros(size(w2)) for i=1:nSamp], [zeros(size(w3)) for i=1:nSamp]
  dbh, dby         = [zeros(size(w4)) for i=1:nSamp], [zeros(size(w5)) for i=1:nSamp]
  dhnext           = zeros(Float64,1,nHid)
  dh               = zeros(nHid,1)
  dhraw            = zeros(nHid,1)

  foreach(x,ytrue,1:nSamp) do xs,ys,s

    fill!(dhnext,0.0);fill!(dh,0.0);fill!(dhraw,0.0)

    nTimes = size(xs,2)
    ypred  = zeros(nTimes)
    hidden = zeros(1, nHid,nTimes+1)
    # Run the prediction
    RNN_predict_loop(nTimes,nVar,nHid,hidden,hidden_pre,hidprew2,ypred,xcur,xs,xw1,w1,w2,w3,w4,w5)

    # Run the derivative backward pass
    for i=nTimes:-1:1

      # Derivative of the loss function
      dy = deriv(lossFunc,ys,ypred,i)

      # Derivative of the activations function (still a scalar)
      dy2 = derivActivation(ypred[i],dy)

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
      @inbounds for j=1:nVar xcur[j]=xs[j,i] end

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


function RNN_predict_loop(nTimes,nVar,nHid,hidden,hidden_pre,hidprew2,ypred,xcur,xs,xw1,w1,w2,w3,w4,w5)

    #Emtpy temporary arrays
    fill!(hidden,zero(eltype(hidden)))
    fill!(hidden_pre,zero(eltype(hidden_pre)))

    for i=1:nTimes

      # First copy current x variables to xcur
      for j=1:nVar xcur[j]=xs[j,i] end

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
      ypred[i] = sigm(dot(hidden_pre,w3) + w5[1])
    end

end


function predict_old(model::RNNModel,w, x) ### This implements w as a vector

  nHid = Int(-0.5*(nVar + 2) + sqrt(0.25*(nVar+2)^2-1+length(w)))
  @reshape_weights(w1=>(nHid,nVar),  w2=>(nHid,nHid), w3=>(nHid,1), w4=>(1,nHid), w5=>(1,))

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
