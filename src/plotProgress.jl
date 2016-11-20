import Plots: plot, gr
export plotSummary
function plotSignal(y)
    gr()
    epIdx=1:length(y[1])
    plot([epIdx, y[3], epIdx, [0,1]],[y[1],y[4], y[2],[0,1]], line=[:line :scatter :line :line], color=["blue" "black" "orange" "red"],markeralpha=0.05, layout=2,fmt=:png)
end

function plotSummary(model)
      gr()
      yNorm=model.yNorm
      pred=predict(model,model.weights,model.xNorm)
      display(plot([model.lossesTrain, model.lossesVali],color=["blue" "orange"],fmt=:png))
      display(plot(vec(pred), vec(yNorm), line=:scatter, color=:black, markeralpha=0.05,fmt=:png))
      println("Correlation: ", cor(vec(pred), vec(yNorm)))
end
