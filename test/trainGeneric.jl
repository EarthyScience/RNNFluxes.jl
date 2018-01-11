function my_generic_model{T}(model,w::AbstractArray{T},x)

    a = w[1]
    b = w[2]
    c = w[3]

    map(x) do xx
        cumsum(a * xx[1,:])+exp.(b * xx[2,:])+cumprod(c * xx[3,:])
    end
end

nTime=100
nSample=200
nVarX = 3

onegood = false

for i=1:10
  x = [rand(nVarX,nTime) for i=1:nSample]

  true_model(x) = transpose(cumsum(x[1,:])+exp.(x[2,:])+cumprod(1.7*x[3,:]))

  y = true_model.(x);

  w      = rand(3) #Init weights
  m=RNNFluxes.GenericModel((),w,my_generic_model)
  train_net(m,x,y,2001);

  xtest = [rand(nVarX,nTime) for i=1:100]
  ytest = true_model.(xtest)
  ypred = predict_after_train(m,xtest)
  if mean(cor.(map(transpose,ytest),ypred)) > 0.999
    onegood=true
    break
  end
end

@test onegood
