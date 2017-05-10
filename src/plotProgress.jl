import Plots: plot, gr, default
export plotSummary
default(show=:inline)
function plotSignal(outputsteps,lossesTrain, lossesVali, predTrain, yTrain,predVali,yVali)
  gr()
  epIdx=outputsteps

  mi,ma = extrema([yTrain;yVali])
  plot([epIdx, predVali, epIdx,   epIdx, [mi,ma], epIdx, epIdx,    predTrain],
		  [lossesTrain,yVali, lossesTrain,    lossesVali,  [mi,ma],  lossesVali,  lossesTrain,    yTrain],
		line=[:line :scatter :line              :line :line  :line                :line    :scatter],
		color=["blue" "orange" "blue"          "orange" "red" "orange"            "blue"   "blue"],
		leg=[true false false],
		markeralpha=0.15, label=["Training" "OBS vs. MOD" "Training" "Validation" "1:1-line" "Validation" "Training" "OBS vs. MOD"],
		layout=@layout([a b; c]),fmt=:png)
end

function plotSummary(model)
      gr()
      yNorm=model.yNorm
      pred=predict(model,model.weights,model.xNorm)
	  #Plot whole training trace
      display(plot([model.lossesTrain, model.lossesVali],color=["blue" "orange"],fmt=:png, label=["Training" "Validation"]))
	  #Plot only last 80% and no peaks
	  nEpoch=length(model.lossesTrain)
	  start=trunc(Int,nEpoch*0.2)
	  idx=start:nEpoch
	  upperQuantTrainLoss=quantile(model.lossesTrain[idx], 0.95)
	  upperQuantValiLoss=quantile(model.lossesVali[idx], 0.95)

	  miniTrainLoss=minimum(model.lossesTrain[idx])
	  miniValiLoss=minimum(model.lossesVali[idx])
      display(plot([idx,idx],[model.lossesTrain[idx], model.lossesVali[idx]],color=["blue" "orange"], fmt=:png, ylim=[(miniTrainLoss, upperQuantTrainLoss) (miniValiLoss,upperQuantValiLoss)], label=["Training" "Validation"], layout=(2,1)))

      #Scatterplot MOD vs OBS
	  min=minimum(vec(yNorm))
	  max=maximum(vec(yNorm))

      display(plot([vec(pred), [min, max]], [vec(yNorm), [min, max]], line=[:scatter :line], color=[:black :red], markeralpha=0.05, label=["OBS vs. MOD", "1:1-line"], fmt=:png))
      println("Correlation: ", cor(vec(pred), vec(yNorm)))
end
