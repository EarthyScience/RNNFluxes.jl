# Example loss functions

For training a model well, it is essential to provide a suitable loss function, which in many cases is more complicated than a simple MSE. To give some inspiration, here are a few examples of loss functions and derivatives that we have used in our work. The function signature of the loss function should be: `(w::Vector, x::Vector{Matrix},y::Vector{Matrix},model::FluxModel)->Number`, where `w` is the weight vector, `x` the predictors, `y` the true targets and model is the model applied. The function returns a single number. The derivative is defined by overloading the `RNNFluxes.deriv` function with a new method depending on the loss function type. The signature is `(t::Type{Function},ytrue,ypred,i)->Number` where `t` is the type of the loss function, `ytrue` is a time series of observations, `ypred` a predicted time series and `i` the index for which the derivative is calculated.

## MSE with missing values

In case there are gaps in the target values, they can be omitted from MSE calculation. We assume here that gaps are encoded as `NaN`s.

````julia
function mselossMiss(w, x, y, model)
    p = RNNFluxes.predict(model,w,x)
    n = mapreduce(i->sum(a->!isnan(a),i),+,y)
    return mapreduce(ii->sumabs2(iix-iiy for (iix,iiy) in zip(ii...) if !isnan(iiy)),+,zip(p,y))/n
end
RNNFluxes.deriv(::typeof(mselossMiss),ytrue::Vector,ypred::Vector,i::Integer)=isnan(ytrue[i]) ? zero(ytrue[i]-ypred[i]) : ytrue[i]-ypred[i]
````

## Mixing loss for each single time step and difference of aggregated fluxes for a full year

As a more complex example, here we combine a normal MSE loss function with one that computes the loss only aggregated on annual values. The factor α determines the ratio to which the individual vs the annual loss are weighted.

````julia
function mselossMissYear(w, x, y, model)
    p = RNNFluxes.predict(model,w,x)
    #Do the annual aggregation
    lossesPerSite = map(y,p) do ytrue,ypred
        nY = length(ytrue)÷NpY
        lossTot = zero(eltype(ypred))
        nTot    = 0
        for iY = 1:nY
            s,sp,n = annMeanN(ytrue,ypred,NpY,iY)
            lossTot += (s - sp)^2
            nTot += n
        end
        lossTot,nTot
    end
    lossAnnual = sum(i->i[1],lossesPerSite)/sum(i->i[2],lossesPerSite)
    nSingle = mapreduce(i->sum(a->!isnan(a),i),+,y)
    lossSingle = mapreduce(ii->sum(abs2(iix-iiy) for (iix,iiy) in zip(ii...) if !isnan(iiy)),+,zip(p,y))/nSingle
    return α * lossAnnual + (1-α) * lossSingle
end



````
