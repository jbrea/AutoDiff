# Standard rectlin layer: f(A,x,b)=rectlin(A*x+b)
# Note that the bias (a column vector) gets expanded here to match the dimension of A*x

FabsAXplusBias(A,X,b)=begin; a=A*X+b*ones(1,size(X,2)); return (abs(a),sign(a)); end

function FabsAXplusBias_inplace(value,aux,A,X,b)
    a=A*X+b*ones(1,size(X,2))
    copy!(value,abs(a))
    copy!(aux,sign(a))
end

function DabsAXplusBias(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,X,b)
    if derivativeIDX==1
        axpy!(1.0,(grad_c.*faux_c)*X',grad_n)
    elseif derivativeIDX==2
        axpy!(1.0,A'*(grad_c.*faux_c),grad_n)
    elseif derivativeIDX==3
        axpy!(1.0,sum(grad_c.*faux_c,2),grad_n)
    end
end

if PROC=="GPU" 

    function FabsAXplusBias_inplace(value,aux,A::CudaArray,X::CudaArray,b::CudaArray)
        FAXplusBias_inplace(value,aux,A,X,b)
        copy!(aux,value)
        abs!(value,value)
    end

    function DabsAXplusBias(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,X::CudaArray,b::CudaArray)
        tmp=CudaArray(Float64,size(grad_c)); fill!(tmp,0.0)
        xsigny_update!(grad_c,faux_c,tmp)
        if derivativeIDX==1
            gemm!('N','T',1.0,tmp,X,1.0,grad_n)
        elseif derivativeIDX==2
            gemm!('T','N',1.0,A,tmp,1.0,grad_n)
        elseif derivativeIDX==3
            ons=CudaArray(Float64,(size(X,2),1)); fill!(ons,1.0)
            gemm!('N','N',1.0,tmp,ons,1.0,grad_n)
            free(ons)
        end
        free(tmp)
    end

end


Derivative[FabsAXplusBias]=DabsAXplusBias
Inplace[FabsAXplusBias]=FabsAXplusBias_inplace

absAXplusBias(A,X,b)=ADnode(FabsAXplusBias,[A X b])
export absAXplusBias


