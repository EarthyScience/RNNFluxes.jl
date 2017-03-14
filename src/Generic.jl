import Base.LinAlg: gemv!
import AutoGrad.grad
using Distributions

"""
    type GenericModel

Type wrapping a generic model that is differentiated with AutoGrad. Only a predict function is needed.
"""
type GenericModel{T,V} <: FluxModel
  weights::Vector{Float64}
  params::T
  lossFunc::Function
  predictFunc::V
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

function GenericModel(params, N::Int, predictFunc; dist = Uniform)
  w=iniWeights(GenericModel, weights, dist)
  GenericModel(w,params,identity,predictFunc,Int[],Float64[],Float64[],NaN,NaN,[zeros(0,0)],[zeros(0,0)],Float64[],Float64[])
end

function GenericModel(params, w, predictFunc)
  GenericModel(w,params,identity,predictFunc,Int[],Float64[],Float64[],NaN,NaN,[zeros(0,0)],[zeros(0,0)],Float64[],Float64[])
end

function iniWeights(::Type{GenericModel}, N, dist)
  weights = rand_flt(0.1, dist, N)  ## Input block
end

function predict(model::GenericModel,w,x)
  model.predictFunc(model,w,x)
end

function predict_with_gradient(model::GenericModel,w, x,ytrue,lossFunc) ### This implements w as a vector
  gradfun = grad(lossFunc)
  return gradfun(w,x,ytrue,model)
end
