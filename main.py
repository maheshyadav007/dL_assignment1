import numpy as np
import wandb
from keras.datasets import fashion_mnist


class NeuralNet:

    nHiddenLayers = 0

    @classmethod
    def update(cls, nHiddenLayers):
        cls.nHiddenLayers = (nHiddenLayers)
    def __init__(self):
        pass
    def __init__(self, nHiddenLayers,neurons,weights ,bias):
        self.nHiddenLayers = nHiddenLayers
        NeuralNet.update(nHiddenLayers)
        self.neurons = neurons
        self.weights = weights
        self.bias = bias
    

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
classLabels = {"0":"T-shirt/top","1":"Trouser","2":"Pullover","3":"Dress","4":"Coat","5":"Sandal","6":"Shirt","7":"Sneaker","8":"Bag","9":"Ankle boot"}

def outEachClass():
    data = []
    labels = []
    for i in range(len(x_train)):
        if classLabels[str(y_train[i])] not in labels:
            data.append(x_train[i])
            labels.append(classLabels[str(y_train[i])])
    print(len(data))
    #wandb.log({"Categories": [wandb.Image(image, caption=caption) for image, caption in zip(data,labels)]})

#------setting up input--------
#Global Variables

n,l,w = x_train.shape
nFeatures = l*w
X = np.reshape(x_train,(n,l*w,1))/255
Y = np.reshape(y_train,(n,1))
weights = []
biases = []
nHiddenLayers = 0
neurons = []
#neuralNet = NeuralNet()

#--------------------------------

def initializeParams(nFeatures,weights,bias,nHiddenLayers,neurons):
    n = nFeatures
    for i in range(nHiddenLayers+1):
        weights.append(np.random.default_rng().uniform(low = -1,high = 1,size = (neurons[i],n)))#rand(neurons[i],n))
        bias.append(np.random.default_rng().uniform(low = -1, high=1, size=(neurons[i],1)))#rand(neurons[i],1))
        n = neurons[i]

def initializeNN(nHiddenLayers,neurons,outputNeurons): 
    weights = []
    bias = []
    neurons.append(outputNeurons)
    initializeParams(nFeatures,weights,bias,nHiddenLayers,neurons)
    return weights, bias

def preActivate(x,w,b):
    out = np.dot((w),x)+b                       
    return out
def logisticActivation(parameter):
    out = (1/(1+np.exp(-1*parameter)))
    return out
def softmaxActivation(parameter):
    exp = np.exp(parameter)
    out = exp/np.sum(exp)
    return out
def feedForwardNN(x,weights,bias):
    nHiddenLayers = len(bias)-1
    a = []
    h = []
    prev = x
    for i in range(nHiddenLayers):
        pA = preActivate(prev,weights[i],bias[i])
        activation = logisticActivation(pA)
        prev = activation
        a.append(pA)
        h.append(activation)
    pA = preActivate(prev,weights[nHiddenLayers],bias[nHiddenLayers])
    a.append(pA)
    output = softmaxActivation(pA)
    return a,h,output



def computeSoftMaxLoss(yHat,y):
    return -1* np.log(yHat[y])

def derivativeActivation(activation,parameter):
    a = 0
    if activation == "logistic":
        a = logisticActivation(parameter)
        a = np.multiply(a,(1-a))
    return a

def backPropagation(a,h,x,y,yHat,weights,nHiddenLayers):
    weightsGrad = [0]*(nHiddenLayers+1)
    biasGrad = [0]*(nHiddenLayers+1)
    el = np.zeros(shape = (10,1))
    el[y] = 1
    aLGrad = -(el-yHat)
    for k in range(nHiddenLayers,0,-1):
        weightsGrad[k] = (np.dot(aLGrad,np.transpose(h[k-1])))
        biasGrad[k] = (aLGrad)
        hk = np.dot(np.transpose(weights[k]),aLGrad)                         #NOTE : hk is really hk-1
        aLGrad = np.multiply(hk,derivativeActivation("logistic",a[k-1]))          
    weightsGrad[0] = (np.dot(aLGrad,np.transpose(x)))
    biasGrad[0] = (aLGrad)
    return weightsGrad,biasGrad


def stochasticGradientDescent(batchSize):
    
    eta = config.learning_rate
    maxIterations = config.epochs
    weights, bias = initializeNN(nHiddenLayers,neurons,outputNeurons)
    weightsGrad, biasGrad = 0, 0
    t = 0
    while t < maxIterations:
        loss = 0
        for x,y in zip(X,Y):
            a,h,yHat = feedForwardNN(x,weights,bias)
            loss += computeLoss(yHat,y)
            wGrad, bGrad = backPropagation(a,h,x,y,yHat,weights)
            if t==0:
                weightsGrad = wGrad
                biasGrad = bGrad
            else :
                weightsGrad += np.array(wGrad)
                biasGrad += np.array(bGrad)

        print(loss)
        wandb.log({'loss':loss})
        #print(weights, weightsGrad)
        weights = weights - np.multiply(eta,weightsGrad)
        bias = bias - np.multiply(eta,biasGrad)

        t+=1





sweep_config = {
    'name' : 'Working sweep',
    "method": "grid",
    'metric': { 
        'name':'loss',
        'goal': 'minimize',
        },
    'parameters':{
        'learning_rate': {'values' : [.001, 0.0001]},
        'epochs' : {'values' : [5,10]},
        'optimizer':{'values' : ['sgd','mgd']},
        'n_hidden_layers' : {'values' : [3,4,5]},
        'size_hidden_layers' : {'values' : [32,64,128]},
        'batch_size' : {'values': [16,32,64]}
        }
}
'''
sweep_config = {
    'name' : 'MY new sweep',
    "method": "grid",
    'metric': { 
        'name':'loss',
        'goal': 'minimize',
        },
    'parameters':{
        'learning_rate': {'values' : [.001, 0.0001]},
        'epochs' : {'values' : [5,10]},
        'optimizer':{'values' : ['sgd']},
        'n_hidden_layers' : {'values' : [3]},
        'size_hidden_layers' : {'values' : [32]},
        'batch_size' : {'values': [64]}
        }
}
'''
hyperparameter_defaults = dict(
    batch_size = 128,
    learning_rate = 0.0001,
    epochs = 5,
    n_hidden_layers = 3,
    size_hidden_layers = 32,
    optimizer = "sgd"
    )


def validate(yHat, y):
    if np.argmax(yHat) == y[0]:
        return True
    return False

def getNextBatch(batchSize,startIndex,notDone):
    Xb,Yb = [], []
    if startIndex+batchSize <= n :
        Xb = X[startIndex:startIndex+batchSize,:]
        Yb = Y[startIndex:startIndex+batchSize, :]
    else :
        Xb = X[startIndex:-1,:]
        Yb = Y[startIndex:-1,:]
        notDone = False
    startIndex = startIndex+batchSize
    return Xb,Yb, startIndex , notDone


def subRoutine(neuralNet,weights,bias,batchSize,startIndex,notDone,lossFunction):
    #weights = neuralNet.weights
    #bias = neuralNet.bias
    nHiddenLayers = neuralNet.nHiddenLayers
    #weightsGrad = weightsGrad
    #biasGrad = biasGrad
    weightsGrad = [0]*( (nHiddenLayers )+ 1)
    biasGrad = [0]*( nHiddenLayers+1)
    nCorrectPredictions = 0
    loss = 0

    Xb,Yb, startIndex, notDone = getNextBatch(batchSize,startIndex,notDone)
    for x,y in zip(Xb,Yb):
        a,h,yHat = feedForwardNN(x, weights, bias)
        
        if validate(yHat,y) :
            nCorrectPredictions += 1
        if lossFunction == "SoftMax":
            loss += computeSoftMaxLoss(yHat,y)
        
        #Computing Gradients
        wGrad, bGrad = backPropagation(a,h,x,y,yHat, weights, nHiddenLayers)# check y bias is not used

        weightsGrad = [ w + wG/ batchSize  for w, wG in zip(weightsGrad, wGrad)]
        biasGrad = [ b + bG/ batchSize for b, bG in zip(biasGrad, bGrad)]
    return startIndex, notDone, weightsGrad, biasGrad, nCorrectPredictions,loss

def train(neuralNet,batchSize,optimizer,lossFunction,eta):
    nHiddenLayers = neuralNet.nHiddenLayers
    startIndex = 0
    notDone = True
    #weightsGrad = [0]*( (nHiddenLayers )+ 1)
    #biasGrad = [0]*( nHiddenLayers+1)
    loss = 0
    nCorrectPredictions = 0
    #updateWeights = [0]*( nHiddenLayers+1)
    #updateBias = [0]*( nHiddenLayers+1)
    while notDone:
        startIndex,notDone,weightsGrad,biasGrad,nCorrect,l = subRoutine(neuralNet,neuralNet.weights,neuralNet.bias,batchSize,startIndex,notDone,lossFunction)
        nCorrectPredictions += nCorrect
        loss += l
            #loss += l

        #global weights = [ w - eta*wG  for w, wG in zip(global weights,weightsGrad)]
        #global bias = [ b - eta*bG  for b, bG in zip(global bias,biasGrad)]
        if optimizer == "sgd":
            #stochasticGradientDescent(neuralNet,weightsGrad,biasGrad,eta)
            Optimizers.SGD(neuralNet,weightsGrad,biasGrad,eta)
        elif optimizer == "mgd":
            #momentumGradientDescent(neuralNet,weightsGrad,biasGrad)
            Optimizers.MGD(neuralNet,weightsGrad,biasGrad,eta)
            #momentumGradientDescent()
        elif optimizer == "nagd":
            if notDone:
                param = (batchSize,startIndex,notDone,lossFunction)
                Optimizers.NAGD(neuralNet,weightsGrad,biasGrad,eta,param)

    return loss/ n, nCorrectPredictions/ n

            
'''
def stochasticGradientDescent(neuralNet,weightsGrad,biasGrad,eta):
    neuralNet.weights = [ w - eta*wG  for w, wG in zip( neuralNet.weights,weightsGrad)]
    neuralNet.bias = [ b - eta*bG  for b, bG in zip( neuralNet.bias,biasGrad)]

def momentumGradientDescent(neuralNet,weightsGrad,biasGrad,eta):
    gamma = 0.9
    updateWeights = [gamma*u + eta*wG/ n  for wG, u in zip(weightsGrad,updateWeights)]
    updateBias = [gamma*u + eta*bG/ n  for bG, u in zip(biasGrad,updateBias)]
    neuralNet.weights = [ w - u  for w, u in zip( neuralNet.weights,updateWeights)]
    neuralNet.bias = [ b - u  for b, u in zip( neuralNet.bias,updateBias)]
'''
class Optimizers:
    #SGD
    def SGD(neuralNet,weightsGrad,biasGrad,eta):
        neuralNet.weights = [ w - eta*wG  for w, wG in zip( neuralNet.weights,weightsGrad)]
        neuralNet.bias = [ b - eta*bG  for b, bG in zip( neuralNet.bias,biasGrad)]
    
    #MGD
    updateWeights = [0]*( NeuralNet.nHiddenLayers+1)
    updateBias = [0]*( NeuralNet.nHiddenLayers+1)
    gamma = 0.9
    def MGD(neuralNet,weightsGrad,biasGrad,eta):
        Optimizers.updateWeights = [Optimizers.gamma*u + eta*wG  for wG, u in zip(weightsGrad,Optimizers.updateWeights)]
        Optimizers.updateBias = [Optimizers.gamma*u + eta*bG  for bG, u in zip(biasGrad,Optimizers.updateBias)]
        neuralNet.weights = [ w - u  for w, u in zip( neuralNet.weights,Optimizers.updateWeights)]
        neuralNet.bias = [ b - u  for b, u in zip( neuralNet.bias,Optimizers.updateBias)]
        
    #NAGD
    #updateWeights = [0]*( NeuralNet.nHiddenLayers+1)
    #updateBias = [0]*( NeuralNet.nHiddenLayers+1)
    #gamma = 0.9
    def NAGD(neuralNet,weightsGrad,biasGrad,eta,param):
        weightsLookahead = [ w - Optimizers.gamma*u  for w, u in zip( neuralNet.weights,Optimizers.updateWeights)]
        biasLookahead = [ b - Optimizers.gamma*u  for b, u in zip( neuralNet.bias,Optimizers.updateBias)]

        batchSize = param[0]
        startIndex = param[1]
        notDone = param[2]
        lossFunction = param[3]
        _,_,weightsLookaheadGrad,biasLookaheadGrad,_,_ = subRoutine(neuralNet,weightsLookahead,biasLookahead,batchSize,startIndex,notDone,lossFunction)

        Optimizers.updateWeights = [Optimizers.gamma*u + eta*wG  for wG, u in zip(weightsLookaheadGrad,Optimizers.updateWeights)]
        Optimizers.updateBias = [Optimizers.gamma*u + eta*bG  for bG, u in zip(biasLookaheadGrad,Optimizers.updateBias)]
        neuralNet.weights = [ w - u  for w, u in zip( neuralNet.weights,Optimizers.updateWeights)]
        neuralNet.bias = [ b - u  for b, u in zip( neuralNet.bias,Optimizers.updateBias)]

    #init    
    def __init__(self):
        Optimizers.updateWeights = [0]*( NeuralNet.nHiddenLayers+1)
        Optimizers.updateBias = [0]*( NeuralNet.nHiddenLayers+1)





def run():
    '''
    wandb.init(config=hyperparameter_defaults ,project="dl_assignment1",entity = "-my")
    
    config = wandb.config
    nHiddenLayers = config.n_hidden_layers
    neurons = []
    neurons.append(config.size_hidden_layers)
    neurons = neurons*nHiddenLayers
    eta = config.learning_rate
    epochs = config.epochs
    batchSize = config.batch_size
    optimizer = config.optimizer

    global nHiddenLayers = nHiddenLayers
    global neurons = neurons
    '''
    #----------
    nHiddenLayers = 3
    neurons = []
    neurons.append(64)
    neurons = neurons*nHiddenLayers
    eta = 0.001
    epochs = 10
    batchSize = 64
    optimizer = "nagd"
    #----------------

    outputNeurons = 10
    lossFunction = "SoftMax"
    weights,bias = initializeNN(nHiddenLayers,neurons,outputNeurons)

    neuralNet = NeuralNet(nHiddenLayers,neurons,weights,bias)
    Optimizers()
    t = 1
    while t <= epochs:
        loss, accuracy = train(neuralNet, batchSize,optimizer,lossFunction,eta)     
        print("After epoch : ",t,"Loss  = ",loss, " Accuracy : ", accuracy)
        #wandb.log({'epoch':epochs,'Loss':loss, "Accuracy":accuracy})
        #wandb.log({"metric":loss })
        t+=1

#sweepId = wandb.sweep(sweep_config,entity = "-my",project = "dl_assignment1")
#wandb.agent(sweepId,function=run)

run()

