import Base.LinAlg: gemv!
import AutoGrad.grad
using Distributions
using ForwardDiff

immutable GradAuto end
immutable GradForward end

"""
    type GenericModel

Type wrapping a generic model that is differentiated with AutoGrad. Only a predict function is needed.
"""
type GenericModel{T,V,G} <: FluxModel
  weights::Vector{Float64}
  params::T
  lossFunc::Function
  predictFunc::V
  gradmethod::G
  outputtimesteps::Vector{Int}
  lossesTrain::Vector{Float64}
  lossesVali::Vector{Float64}
end

function GenericModel(params, N::Int, predictFunc; dist = Uniform, diff="forward")
  w=iniWeights(GenericModel, weights, dist)
  GenericModel(w,params,identity,predictFunc,diff=="forward" ? GradForward() : GradAuto(),Int[],Float64[],Float64[])
end

function GenericModel(params, w, predictFunc; diff="forward")
  GenericModel(w,params,identity,predictFunc,diff=="forward" ? GradForward() : GradAuto(),Int[],Float64[],Float64[])
end

function iniWeights(::Type{GenericModel}, N, dist)
  weights = rand_flt(0.1, dist, N)  ## Input block
end

function predict(model::GenericModel,w,x)
  model.predictFunc(model,w,x)
end

function predict_with_gradient{T,U}(model::GenericModel{T,U,GradAuto},w, x,ytrue,lossFunc) ### This implements w as a vector
  gradfun = grad(lossFunc)
  return gradfun(w,x,ytrue,model)
end

using ForwardDiff
function predict_with_gradient{T,U}(model::GenericModel{T,U,GradForward},w, x,ytrue,lossFunc) ### This implements w as a vector
  return ForwardDiff.gradient(we->lossFunc(we,x,ytrue,model),w)
end

function normalize_data(model::GenericModel,x,y)
  x,y
end

function normalize_data_inv(model::GenericModel,y)
  y
end
