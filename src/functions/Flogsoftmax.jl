function Flogsoftmax(X::Array{Float64,2}, A::Array{Int64,1})
	res = zeros(1)
	for i in 1:length(A)
		m = maximum(X[:,i])
		res[1] += X[A[i], i] - log(sum(exp(X[:,i] - m))) - m
	end
	return (res/length(X), nothing)
end

function Flogsoftmax_inplace(res, aux, X::Array{Float64,2}, A::Array{Int64,1})
	res[1] = 0.
	for i in 1:length(A)
		m = maximum(X[:,i])
		res[1] += X[A[i], i] - log(sum(exp(X[:,i] - m))) - m
	end
	res[1] *= 1./length(X)
end

function Dlogsoftmax(derivativeIDX,f_c,faux_c,grad_c,grad_n,
					 X::Array{Float64,2}, A::Array{Int64,1})
	#if derivativeIDX == 1
		for i in 1:length(A)
			expx = exp(X[:,i] - maximum(X[:,i]))
			normexpx = (expx ./ sum(expx))
			for a in 1:size(X,1)
				grad_n[a,i] = -normexpx[a]
				if a == A[i]
					grad_n[a,i] += 1.
				end
				grad_n[a,i] *= 1./length(X)
			end
		end
	#end
end

Derivative[Flogsoftmax]=Dlogsoftmax
Inplace[Flogsoftmax]=Flogsoftmax_inplace

logsoftmax(A, X) = ADnode(Flogsoftmax, [X, A])
export logsoftmax
