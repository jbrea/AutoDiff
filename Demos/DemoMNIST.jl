# Training a deep autoencoder on MNIST
# The method uses Nesterov's accelerated gradient, with minibatches
# (c) David Barber, University College London 2015

# Note that this Float64 version is quite slow -- see the DemoMNIST32 for a faster Float32 version

PlotResults=true

#useproc("GPU") 
useproc("CPU") # GPU about 4 times faster than CPU

using MAT

Ntrain=1000
BatchSize=1000
TrainingIts=100 # number of Nesterov updates
include("loadmnist.jl")
images,label=loadmnist()
r=randperm(size(images,2))
data=images[:,r]
# bound away from 0 and 1 to avoid log(0) problems:
tol=0.000001
data[data.>(1-tol)]=1-tol
data[data.<tol]=tol

H=[784 1000 500 250 30 250 500 1000 784] # number of units in each layer

L=length(H) # number of hidden layers
# node indices:
w=Array(ADnode,L) # weights
bias=Array(ADnode,L) # biases
h=Array(ADnode,L) # hidden layer index (input layer is h[1])

StartCode()
ytrain=h[1]=ADnode()
for layer=2:L-1
    w[layer]=ADvariable()
    bias[layer]=ADvariable()
    #h[layer]=kinklinAXplusBias(w[layer],h[layer-1],bias[layer])
#    h[layer]=AXplusBias(w[layer],h[layer-1],bias[layer])
#    h[layer]=absAXplusBias(w[layer],h[layer-1],bias[layer])
    h[layer]=rectlinAXplusBias(w[layer],h[layer-1],bias[layer])
#    h[layer]=abs(w[layer]*h[layer-1])+1.5*w[layer]*h[layer-1]
end
w[L]=ADvariable()
bias[L]=ADvariable()
h[L]=AXplusBias(w[L],h[L-1],bias[L])
ypred=sigmoid(h[L]) # just use for testing, not training
meanSqloss=meanSquareLoss(ypred,ytrain)
loss=BinaryKullbackLeiblerLossXsigmoidY(ytrain,h[L])
net=EndCode()

#instantiate root node values:
net.value[h[1]]=data[:,1:BatchSize]

for i=2:L
    net.value[w[i]]=.5*sign(randn(H[i],H[i-1]))/sqrt(H[i-1])
    net.value[bias[i]]=.005*sign(randn(H[i],1))+0.1
end
initvalue=deepcopy(net.value[w[L]])

#convert!(Array{Float32},net.value)
net=compile(net) # compile the DAG and preallocate memory



@gpu CUDArt.init([0]) # let the user do device management
@gpu net=convert(net,PROC)


#gradcheck(net)

    println("Training: using $(net.gpu==true? "GPU" : "CPU") $(net.eltype)")
tstart=time()
error=Array(Float64,0)
ParsToUpdate=Parameters(net)
velo=NesterovInit(net)
#avgrad=GradientDescentMomentumInit(net); Momentum=0.5
minibatchstart=1 # starting datapoint for the minibatch
#ForwardPassList!(net,ExcludeNodes=[ypred])
for iter=1:TrainingIts
    LearningRate=0.2/(1+iter/5000)
    minibatchstart,minibatch=GetBatch(minibatchstart,BatchSize,Ntrain)
    net.value[ytrain]=cArray(PROC,data[:,minibatch]) # select batch of data
    #net.value[ytrain]=cArray(data[:,minibatch]) # select batch of data
    ADforward!(net)
    ADbackward!(net)
    push!(error,extract(net.value[net.FunctionNode])[1])
    printover("iteration $iter: training loss = $(error[iter]) : meanSqLoss = $(784*net.value[meanSqloss])")
    for par in ParsToUpdate
        #GradientDescentUpdate!(net.value[par],net.gradient[par],LearningRate)
        #GradientDescentMomentumUpdate!(net.value[par],net.gradient[par],avgrad[par],Momentum,LearningRate)
        NesterovGradientDescentUpdate!(net.value[par],net.gradient[par],velo[par],LearningRate,iter)
    end
end
tend=time();
println("\nTraining took $(tend-tstart) seconds\n")

if PlotResults
    net=convert(net,"CPU") # to make analysis easier
    figure(1)
    plot(error); title("training loss")
    figure(2)
    for i=1:10 # plot the reconstructions for a few datapoints
        p=imshow([reshape(net.value[h[1]][:,i],28,28)'  reshape(net.value[ypred][:,i],28,28)'],interpolation="none",cmap=ColorMap("gray"))
        display(p)
        println("press key to continue")
        readline(STDIN)
    end
end

