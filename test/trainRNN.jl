df = readtable(joinpath(Pkg.dir("RNNFluxes"),"data","MLTestSeasonal2.csv"), header=true)
#describe(df)
#head(df)
xOrig = convert(Array, df[[:x_1, :x_2, :x_3]])
# Predictor has dimensions (time, sample, variables)
xOrig = reshape(xOrig, 100, 500, 3)
subsample=1:10:500
x = xOrig[:,subsample,:] ## For performance only choose fewer samples (sites), to be fastest do x = xOrig[:,1:1,:] , but batchsize must be 1
### but real-world case will have >500 samples!!!! Even >>1000 (e.g. number of pixels in ESDC....)
yOrig = convert(Array, df[:yOBS])
yOrig = reshape(yOrig, 100, 500)
y = yOrig[:, subsample, :] ## For performance only choose fewer samples (sites)

srand(1000)
w=iniWeights(3, 12, "RNN")
trainResult = train_net(x, y, 201, weights=w, nHid=12, batchSize=5, infoStepSize=100, plotProgress=false)
predFunc,  w, lossesTrain, lossesVali, (yMin, yMax), xEx =trainResult
