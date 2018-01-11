nTime=100
nSample=200
nVarX = 3

onegood = false

for i=1:10
  x = [rand(nVarX,nTime) for i=1:nSample]

  true_model(x) = transpose(cumsum(x[1,:])+exp.(x[2,:])+cumprod(1.7*x[3,:]))

  y = true_model.(x);

  nHid=10
  m=RNNFluxes.RNNModel(nVarX,nHid)
  train_net(m,x,y,10001);

  xtest = [rand(nVarX,nTime) for i=1:100]
  ytest = true_model.(xtest)
  ypred = predict_after_train(m,xtest)
  if mean(cor.(map(transpose,ytest),ypred)) > 0.85
    onegood=true
    break
  end
end

@test onegood
