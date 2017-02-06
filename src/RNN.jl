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
  xMin::Vector{Float64}
  xMax::Vector{Float64}
end

function RNNModel(nVar,nHid)
  w=iniWeights(RNNModel,nVar,nHid)
  totalNweights=nVar*nHid + nHid * nHid + 2*nHid + 1
  length(w) == totalNweights ||  error("Length of weights $(size(weights,1)) does not match needed length $totalNweights!")
  RNNModel(w,nHid,0,identity,Int[],Float64[],Float64[],NaN,NaN,zeros(0,0,0),zeros(0,0,0),Tuple{Float64,Float64}[])
end



function iniWeights(::Type{RNNModel},nVarX::Int=3, nHid::Int=12)
  weights = -1.0 .+ 2.0 .*
  [rand_flt(-1./sqrt(nVarX), 1./sqrt(nVarX), nVarX, nHid),  ## Input to hidden
  rand_flt(1./-sqrt(nHid), 1./sqrt(nHid),  nHid,nHid),  ## HIdden to hidden
  rand_flt(1./-sqrt(nHid), 1./sqrt(nHid),  nHid, 1), ## Hidden to output
  0.0 * rand(Float64, 1, nHid),  ## Bias to hidden initialized to zero
  0.0 * rand(Float64, 1)] ## bias to output to zero
  weights=raggedToVector(weights)
end

function predict(::RNNModel,w, x) ### This implements w as a vector

  # Reshape weight vectors and create temp arrays
  nTimes,nSamp,nVar,nHid,ypred,w1,w2,w3,w4,w5,hidden_pre,xw1,hidprew2,xcur,hidden = RNN_init_pred(w,x)

  for s=1:nSamp  ## could this loop be parallelized into say 100 parallel instances? Even a speed up factor 10 would make quite some diff
    # Run the prediction
    RNN_predict_loop(s,nTimes,nVar,nHid,hidden,hidden_pre,hidprew2,ypred,xcur,x,xw1,w1,w2,w3,w4,w5)
  end

  return ypred
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


function predict_old(model::RNNModel,w, x) ### This implements w as a vector

  nHid = Int(-0.5*(nVar + 2) + sqrt(0.25*(nVar+2)^2-1+length(w)))
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
