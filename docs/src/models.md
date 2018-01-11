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

A short example on how to define and train such a generic model follows. Let's assume we already know the model structure but want to estimate the parameters. Any Julia code would be valid here in defining this model:

````julia
function my_generic_model{T}(model,w::AbstractArray{T},x)

    a = w[1]
    b = w[2]
    c = w[3]

    map(x) do ix
        cumsum(a * ix[1,:])+exp.(b * ix[2,:])+cumprod(c * ix[3,:])
    end
end
````

We generate some artificial data again:

````julia
nTime=100
nSample=200
nVarX = 3
x = [rand(nVarX,nTime) for i=1:nSample]

true_model(x) = transpose(cumsum(x[1,:])+exp.(x[2,:])+cumprod(1.7*x[3,:]))

y = true_model.(x);
````

In the next step we define a `GenericModel` which includes our above-mentioned predict function.

````julia
params = ()
w      = rand(3) #Init weights
m      = RNNFluxes.GenericModel(params,w,my_ANN_model)
````


Training works "as usual"

````julia
t = train_net(m,x,y,4001);
````
