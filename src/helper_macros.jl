macro reshape_weights(inputs...)
  ex=quote end
  offs_ex = 1
  fin_ex  = 0

  for i=1:length(inputs)
    scur=inputs[i].args[2]
    !isa(scur,Expr) && (scur=Expr(:tuple,scur))
    wsym=esc(inputs[i].args[1])
    mul_sizes = Expr(:call,:*,(scur.args)...)
    fin_ex=:($fin_ex + $mul_sizes)
    reshex = Expr(:call,:reshape,:(w[$(copy(offs_ex)):$fin_ex]),scur.args...)
    push!(ex.args,:($wsym = $reshex))
    offs_ex = :($offs_ex + $mul_sizes)
  end
  ex
end
