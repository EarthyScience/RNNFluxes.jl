function normalize_data(model::FluxModel,x,y)
  ### Normalize y to 0,1

  xNorm,xMin,xMax = normalize_data(model,x,-1.0,1.0)
  model.xNorm,model.xMin,model.xMax = xNorm,xMin,xMax

  if y!=nothing
    yNorm,yMin,yMax = normalize_data(model,y,0.0,1.0)
    model.yNorm,model.yMin,model.yMax = yNorm,yMin[1],yMax[1]
  else
    yNorm=y
  end
  xNorm, yNorm
end

function normalize_data_inv(model::FluxModel,y)
  return map(iy->iy .* (model.yMax-model.yMin) .+ model.yMin,y)
end

function normalize_data(model::FluxModel,x,newmin,newmax)
  xall=hcat(x...)
  xMin,xMax=zeros(size(xall,1)),zeros(size(xall,1))
  for i=1:size(xall,1)
    xv = xall[i,:]
    xMin[i],xMax[i]=minimum(xv[!isnan(xv)]),maximum(xv[!isnan(xv)])
  end
  xNorm=deepcopy(x)
  for (xx,xxNorm) in zip(x,xNorm)
    for j in 1:size(xx,2), v in 1:size(xx,1)
      xxNorm[v,j] = (newmax-newmin).* ((xx[v,j]-xMin[v])/(xMax[v]-xMin[v]))+newmin
    end
  end
  xNorm,xMin,xMax
end
