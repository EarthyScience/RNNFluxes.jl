# Available Models

## Recurrent models with hand-coded gradients

```@docs
RNNFluxes.LSTMModel
```

```@docs
RNNFluxes.RNNModel
```

## Trainning a generic model using AutoDiff

We have also wrapped the training and plotting machinery here together with the ForwardDiff package, so that you can invert any custom model, too. Please note that, for this prupose the package (Flux.jl)[https://github.com/FluxML/Flux.jl] might be a better and well-maintained option.  

```@docs
RNNFluxes.GenericModel
```

A short example on how to define and train such a generic model follows. Let's assume we define an ANN from scratch. Of course this could also be a mechanistic model, or a network coupled to a mechanistic model etc:

````julia
using RNNFluxes

import Knet.sigm
function my_ANN_model{T}(model,w::AbstractArray{T},x)
    nx,n1,n2=model.params
    @reshape_weights(w_In_N1=>(nx,n1), w_Bi_N1=>(1,n1),
                     w_N1_N2=>(n1,n2), w_Bi_N2=>(1,n2),
                     w_N2_Ou=>(n2,)  , w_Bi_Ou=>(1,))
    a = w[end-1]
    b = w[end]

    #Map over samples
    y = map(x) do ix
        yout=Array{T}(size(ix,2),1)
        #Loop over time
        for itime = 1:size(ix,2)
            l1 = sigm.(ix[:,itime]' * w_In_N1 + w_Bi_N1)
            l2 = sigm.(l1* w_N1_N2 + w_Bi_N2)
            yout[itime,1]= sigm.(l2* w_N2_Ou + w_Bi_Ou)[1]
        end
        a+b*yout
    end
    y
end
````

We generate some artificial data again:

````julia
nSample = 100
nX      = 4
nTimest = 50
ftrue = x->cos.(sum(exp.(x),1))

xdata = [randn(nX,nTimest) for i=1:nSample]
ydata = map(ftrue,xdata);
````

In the next step we define a `GenericModel` which includes our above-mentioned predict function.

````julia
n1 = 30 #First layer
n2 = 10 #Second layer
params = (nX,n1,n2)
N      = (nX+1)*n1 + (n1+1)*n2 + n2+1 #Number of weights
w      = rand(N)*0.1-0.05 #Init weights
model=RNNFluxes.GenericModel(params,w,my_ANN_model)
````


Training works "as usual"

````julia
t=train_net(model,xdata,ydata,4001);
````
