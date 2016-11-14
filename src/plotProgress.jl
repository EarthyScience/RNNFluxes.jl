"""
     module PlotProgress

Contains some utility functions to monitor the progress of the training. Designed to work in
conjunction with jupyter notebooks, might or mimght not work in other environments.
"""
module PlotProgress
import Plots: plot, gr
function plotSignal(y)
    gr()
    epIdx=1:length(y[1])
    plot([epIdx, y[3], epIdx, [0,1]],[y[1],y[4], y[2],[0,1]], line=[:line :scatter :line :line], color=["blue" "black" "orange" "red"],markeralpha=0.05, layout=2,fmt=:png)
end

function plotSummary(lossesTrain, lossesVali,w,xNorm,yNorm,pred)
      gr()
      display(plot([lossesTrain, lossesVali],color=["blue" "orange"]))
      display(plot(vec(pred), vec(yNorm), line=:scatter, color=:black, markeralpha=0.05))
      println("Correlation: ", cor(vec(pred), vec(yNorm)))
end

end
