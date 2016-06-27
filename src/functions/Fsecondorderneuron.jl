function Fsecondorderneuron(A, B, C, X)
	y = zeros(size(C, 1), size(X, 2))
	for i in 1:size(C, 1)
		for m in 1:size(X, 2)
			y[i, m] += C[i]
			for j in 1:size(X, 1)
				y[i, m] += B[i, j] * X[j, m]
				for k in 1:size(X, 1)
					y[i, m] += A[i, j, k] * X[j, m] * X[k, m]
				end
			end
		end
	end
	return (y, nothing)
end

function Fsecondorderneuron_inplace(y, aux, A, B, C, X)
	for i in 1:size(C, 1)
		for m in 1:size(X, 2)
			y[i, m] = C[i]
			for j in 1:size(X, 1)
				y[i, m] += B[i, j] * X[j, m]
				for k in 1:size(X, 1)
					y[i, m] += A[i, j, k] * X[j, m] * X[k, m]
				end
			end
		end
	end
end

function Dsecondorderneuron(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,B,C,X)
	if derivativeIDX == 1
		for i in 1:size(A,1)
			for j in 1:size(A,2)
				for k in 1:size(A,3)
					for m in 1:size(X,2)
						grad_n[i,j,k] += grad_c[i,m] * X[j,m] * X[k,m]
					end
				end
			end
		end
	elseif derivativeIDX == 2
		axpy!(1.0,grad_c*X',grad_n)
	elseif derivativeIDX == 3
		axpy!(1.0,sum(grad_c,2),grad_n)
	elseif derivativeIDX == 4
		for j in 1:size(X,1)
			for m in 1:size(X,2)
				for i in 1:size(A,1)
					grad_n[j,m] += grad_c[i,m] * B[i,j]
					for k in 1:size(X,1)
						grad_n[j,m] += grad_c[i,m] * (A[i,k,j] + A[i,j,k]) * X[k,m]
					end
				end
			end
		end
	end
end

Derivative[Fsecondorderneuron]=Dsecondorderneuron
Inplace[Fsecondorderneuron]=Fsecondorderneuron_inplace

secondorderneuron(A, B, C, X) = ADnode(Fsecondorderneuron, [A B C X])
export secondorderneuron

