import numpy as np
import wandb
import tensorflow as tf
from keras.datasets import fashion_mnist


class NeuralNet:

    nHiddenLayers = 0

    @classmethod
    def update(cls, nHiddenLayers):
        cls.nHiddenLayers = (nHiddenLayers)
    def __init__(self):
        pass
    def __init__(self, nHiddenLayers,neurons,weights ,bias,hLActivationFunc,outputActivationFunc,lossFunc):
        self.nHiddenLayers = nHiddenLayers
        NeuralNet.update(nHiddenLayers)
        self.neurons = neurons
        self.weights = weights
        self.bias = bias
        self.hLActivationFunc = hLActivationFunc
        self.outputActivationFunc = outputActivationFunc
        self.lossFunc = lossFunc
    

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
classLabels = {"0":"T-shirt/top","1":"Trouser","2":"Pullover","3":"Dress","4":"Coat","5":"Sandal","6":"Shirt","7":"Sneaker","8":"Bag","9":"Ankle boot"}

inputX = x_train
outputY = y_train
def outEachClass():
    data = []
    labels = []
    for i in range(len(inputX)):
        if classLabels[str(outputY[i])] not in labels:
            data.append(inputX[i])
            labels.append(classLabels[str(outputY[i])])
    wandb.log({"Categories": [wandb.Image(image, caption=caption) for image, caption in zip(data,labels)]})

#------setting up input--------
#Global Variables
n,l,w = x_test.shape
x_test, y_test = np.reshape(x_test,(n,l*w,1))/255 , np.reshape(y_test,(n,1))


n,l,w = x_train.shape
nFeatures = l*w
nTrain = int(.9*n)

x_train, y_train = np.reshape(x_train,(n,l*w,1))/255 , np.reshape(y_train,(n,1))
X, Y = x_train[0:nTrain,:], y_train[0:nTrain,:]
XVal, YVal = x_train[nTrain:n-1,:], y_train[nTrain:n-1,:]

n = nTrain
weights = []
biases = []
nHiddenLayers = 0
neurons = []
#neuralNet = NeuralNet()

#--------------------------------

def initializeParams(nFeatures,weights,bias,nHiddenLayers,neurons,initializer):
    n = nFeatures
    initi = tf.keras.initializers.GlorotNormal()
    for i in range(nHiddenLayers+1):
        if initializer == 'random':
            weights.append(np.random.default_rng().uniform(low = -0.5,high = 0.5,size = (neurons[i],n)))#rand(neurons[i],n))
            bias.append(np.random.default_rng().uniform(low = -0.5, high=0.5, size=(neurons[i],1)))#rand(neurons[i],1))
            n = neurons[i]
        elif initializer == "xavier":
            '''
            std = 2/(n + neurons[i])
            weights.append(np.random.default_rng().normal(loc = 0,scale = std,size = (neurons[i],n)))#rand(neurons[i],n))
            bias.append(np.random.default_rng().normal(low = 0, scale = std, size=(neurons[i],1)))#rand(neurons[i],1))
            n = neurons[i]
            '''
            values = initi(shape=(neurons[i],n))
            weights.append(values)
            values = initi(shape=(neurons[i],1))
            bias.append(values)
            n = neurons[i]

def initializeNN(nHiddenLayers,neurons,outputNeurons,initializer): 
    weights = []
    bias = []
    neurons.append(outputNeurons)
    initializeParams(nFeatures,weights,bias,nHiddenLayers,neurons,initializer)
    return weights, bias

def preActivate(x,w,b):
    out = np.dot((w),x)+b                       
    return out
def activation(activationFunction,parameter):
    out = 0
    if activationFunction == "sigmoid" :
        out = (1/(1+np.exp(-1*parameter)))
    elif activationFunction == "tanh":
        eP ,eN = np.exp(parameter), np.exp(-1*parameter)
        out = ((eP - eN)/(eP + eN))
    elif activationFunction == "relu":
        out = np.maximum(0,parameter)
    elif activationFunction == "softmax":
        exp = np.exp(parameter)
        out = exp/np.sum(exp)
    return out
#def softmaxActivation(parameter):
#   exp = np.exp(parameter)
#    out = exp/np.sum(exp)
#    return out
def feedForwardNN(neuralNet,x,weights,bias):  #do not change arguments
    nHiddenLayers = len(bias)-1
    a = []
    h = []
    prev = x
    for i in range(nHiddenLayers):
        pA = preActivate(prev,weights[i],bias[i])
        activ = activation(neuralNet.hLActivationFunc,pA)
        prev = activ
        a.append(pA)
        h.append(activ)
    pA = preActivate(prev,weights[nHiddenLayers],bias[nHiddenLayers])
    a.append(pA)
    output = activation(neuralNet.outputActivationFunc,pA)
    return a,h,output

def computeLoss(lossFunc,yHat,y):
    if lossFunc == "crossentropy":
        return -1* np.log(yHat[y])[0][0]
    elif lossFunc == "mse":
        e = np.zeros((10,1))
        e[y] = 1
        return np.sum((yHat-y)**2)

def derivativeActivation(activationFunction,parameter):
    a = 0
    if activationFunction == "sigmoid":
        a = activation(activationFunction,parameter)
        a = np.multiply(a,(1-a))
    elif activationFunction == "tanh":
        a = activation(activationFunction,parameter)
        a = 1 - np.multiply(a,a)
    elif activationFunction == "relu":
        a = activation(activationFunction,parameter)
        for i in range(0,a.shape[0]):
            if a[i] > 0 :
                a[i] = 1
    return a

def backPropagation(a,h,x,y,yHat,weights,nHiddenLayers,neuralNet):
    weightsGrad = [0]*(nHiddenLayers+1)
    biasGrad = [0]*(nHiddenLayers+1)
    el = np.zeros(shape = (10,1))
    el[y] = 1
    aLGrad = np.zeros(shape = (10,1))
    if neuralNet.lossFunc == "crossentropy" :
        aLGrad = -(el-yHat)
    elif neuralNet.lossFunc == "mse":
        for i in range(0,10):
            I = np.zeros(shape = (10,1))
            I[i] = 1
            aLGrad[i] = 2* np.sum(np.multiply(np.multiply(yHat-el,yHat),I-yHat[i]))
        
    for k in range(nHiddenLayers,0,-1):
        weightsGrad[k] = (np.dot(aLGrad,np.transpose(h[k-1])))
        biasGrad[k] = (aLGrad)
        hk = np.dot(np.transpose(weights[k]),aLGrad)                         #NOTE : hk is really hk-1
        aLGrad = np.multiply(hk,derivativeActivation(neuralNet.hLActivationFunc,a[k-1]))          
    weightsGrad[0] = (np.dot(aLGrad,np.transpose(x)))
    biasGrad[0] = (aLGrad)
    return weightsGrad,biasGrad

def validate(yHat, y):
    if np.argmax(yHat) == y[0]:
        return True
    return False
def predict(X,Y,neuralNet):
    nCorrectPredictions = 0
    prediction = []
    loss = 0
    for x, y in zip(X,Y):
        _, _, yHat = feedForwardNN(neuralNet,x,neuralNet.weights,neuralNet.bias)
        prediction.append(np.argmax(yHat))
        if neuralNet.lossFunc == "crossentropy":
            loss += computeLoss("crossentropy",yHat,y)
        elif neuralNet.lossFunc == "mse":
            loss += computeLoss("mse",yHat,y)
        if validate(yHat,y):
            nCorrectPredictions += 1
    return loss/X.shape[0], nCorrectPredictions/X.shape[0],prediction

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


def subRoutine(neuralNet,weights, bias, batchSize,startIndex,notDone):  #Do not change arguments 
    #weights = neuralNet.weights
    #bias = neuralNet.bias
    lossFunc = neuralNet.lossFunc
    nHiddenLayers = neuralNet.nHiddenLayers
    #weightsGrad = weightsGrad
    #biasGrad = biasGrad
    weightsGrad = [0]*( (nHiddenLayers )+ 1)
    biasGrad = [0]*( nHiddenLayers+1)
    nCorrectPredictions = 0
    loss = 0

    Xb,Yb, startIndex, notDone = getNextBatch(batchSize,startIndex,notDone)
    for x,y in zip(Xb,Yb):
        a,h,yHat = feedForwardNN(neuralNet,x, weights, bias)
        
        if validate(yHat,y) :
            nCorrectPredictions += 1
        if lossFunc == "crossentropy":
            loss += computeLoss("crossentropy",yHat,y)
        elif lossFunc == "mse":
            loss += computeLoss("mse",yHat,y)
        
        #Computing Gradients
        wGrad, bGrad = backPropagation(a,h,x,y,yHat, weights, nHiddenLayers,neuralNet)# check y bias is not used

        weightsGrad = [ w + wG/ batchSize  for w, wG in zip(weightsGrad, wGrad)]
        biasGrad = [ b + bG/ batchSize for b, bG in zip(biasGrad, bGrad)]
    return startIndex, notDone, weightsGrad, biasGrad, nCorrectPredictions,loss

def train(neuralNet,batchSize,optimizer,eta,alpha):
    nHiddenLayers = neuralNet.nHiddenLayers
    startIndex = 0
    notDone = True
    
    loss = 0
    nCorrectPredictions = 0
    
    while notDone:
        startIndex,notDone,weightsGrad,biasGrad,nCorrect,l = subRoutine(neuralNet,neuralNet.weights,neuralNet.bias,batchSize,startIndex,notDone)
        nCorrectPredictions += nCorrect
        loss += l

        #Regularization
        weightsGrad = [wG + alpha* w/batchSize for w,wG in zip(neuralNet.weights,weightsGrad)]

        if optimizer == "sgd":
            Optimizers.SGD(neuralNet,weightsGrad,biasGrad,eta)
        elif optimizer == "mgd":
            Optimizers.MGD(neuralNet,weightsGrad,biasGrad,eta)
        elif optimizer == "nagd":
            if notDone:
                param = (batchSize,startIndex,notDone,neuralNet.lossFunc)
                Optimizers.NAGD(neuralNet,weightsGrad,biasGrad,eta,param)
        elif optimizer == "rmsprop":
            Optimizers.RMSProp(neuralNet,weightsGrad,biasGrad,eta)
        elif optimizer == "adam":
            Optimizers.Adam(neuralNet,weightsGrad,biasGrad,eta)
        elif optimizer == "nadam":
            Optimizers.Nadam(neuralNet,weightsGrad,biasGrad,eta)

    return loss/ n, nCorrectPredictions/ n


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
        _,_,weightsLookaheadGrad,biasLookaheadGrad,_,_ = subRoutine(neuralNet,weightsLookahead,biasLookahead,batchSize,startIndex,notDone)

        Optimizers.updateWeights = [Optimizers.gamma*u + eta*wG  for wG, u in zip(weightsLookaheadGrad,Optimizers.updateWeights)]
        Optimizers.updateBias = [Optimizers.gamma*u + eta*bG  for bG, u in zip(biasLookaheadGrad,Optimizers.updateBias)]
        neuralNet.weights = [ w - u  for w, u in zip( neuralNet.weights,Optimizers.updateWeights)]
        neuralNet.bias = [ b - u  for b, u in zip( neuralNet.bias,Optimizers.updateBias)]

    #RMSProp
    #updateWeights = [0]*( NeuralNet.nHiddenLayers+1)
    #updateBias = [0]*( NeuralNet.nHiddenLayers+1)
    beta = 0.9
    eps = 1e-8
    def RMSProp(neuralNet,weightsGrad,biasGrad,eta):
        Optimizers.updateWeights = [Optimizers.beta*u + (1-Optimizers.beta)*(wG**2) for wG, u in zip(weightsGrad,Optimizers.updateWeights)]
        Optimizers.updateBias = [Optimizers.beta*u + (1-Optimizers.beta)*(bG**2)  for bG, u in zip(biasGrad,Optimizers.updateBias)]
        neuralNet.weights = [ w - np.multiply((eta/(np.sqrt(u+Optimizers.eps))),wG)  for w,wG, u in zip( neuralNet.weights,weightsGrad,Optimizers.updateWeights)]
        neuralNet.bias = [ b - np.multiply((eta/(np.sqrt(u+Optimizers.eps))),bG)  for b,bG, u in zip( neuralNet.bias,biasGrad,Optimizers.updateBias)]
    
    #Adam
    beta1, beta2, eps ,t= 0.9, 0.999, 1e-8, 1
    mW = [0]*( NeuralNet.nHiddenLayers+1)
    mB = [0]*( NeuralNet.nHiddenLayers+1)
    uW = [0]*( NeuralNet.nHiddenLayers+1)
    uB = [0]*( NeuralNet.nHiddenLayers+1)
    def Adam(neuralNet,weightsGrad,biasGrad,eta):
        Optimizers.mW = [Optimizers.beta1*u + (1-Optimizers.beta1)*(wG) for wG, u in zip(weightsGrad,Optimizers.mW)]
        Optimizers.mB = [Optimizers.beta1*u + (1-Optimizers.beta1)*(bG) for bG, u in zip(biasGrad,Optimizers.mB)]
        Optimizers.uW = [Optimizers.beta2*u + (1-Optimizers.beta2)*(wG**2) for wG, u in zip(weightsGrad,Optimizers.uW)]
        Optimizers.uB = [Optimizers.beta2*u + (1-Optimizers.beta2)*(bG**2) for bG, u in zip(biasGrad,Optimizers.uB)]
        
        mWHat = [m/(1-Optimizers.beta1**Optimizers.t) for m in Optimizers.mW]
        mBHat = [m/(1-Optimizers.beta1**Optimizers.t) for m in Optimizers.mB]
        uWHat = [u/(1-Optimizers.beta2**Optimizers.t) for u in Optimizers.uW]
        uBHat = [u/(1-Optimizers.beta2**Optimizers.t) for u in Optimizers.uB]
        neuralNet.weights = [ w - np.multiply((eta/(np.sqrt(u+Optimizers.eps))),m)  for w,m, u in zip( neuralNet.weights,mWHat,uWHat)]
        neuralNet.bias = [ b - np.multiply((eta/(np.sqrt(u+Optimizers.eps))),m)  for b,m, u in zip( neuralNet.bias,mBHat,uBHat)]
        Optimizers.t += 1
    def Nadam(neuralNet,weightsGrad,biasGrad,eta):
        Optimizers.mW = [Optimizers.beta1*u + (1-Optimizers.beta1)*(wG) for wG, u in zip(weightsGrad,Optimizers.mW)]
        Optimizers.mB = [Optimizers.beta1*u + (1-Optimizers.beta1)*(bG) for bG, u in zip(biasGrad,Optimizers.mB)]
        Optimizers.uW = [Optimizers.beta2*u + (1-Optimizers.beta2)*(wG**2) for wG, u in zip(weightsGrad,Optimizers.uW)]
        Optimizers.uB = [Optimizers.beta2*u + (1-Optimizers.beta2)*(bG**2) for bG, u in zip(biasGrad,Optimizers.uB)]

        mWHat = [m/(1-Optimizers.beta1**Optimizers.t) for m in Optimizers.mW]
        mBHat = [m/(1-Optimizers.beta1**Optimizers.t) for m in Optimizers.mB]
        uWHat = [u/(1-Optimizers.beta2**Optimizers.t) for u in Optimizers.uW]
        uBHat = [u/(1-Optimizers.beta2**Optimizers.t) for u in Optimizers.uB]
        neuralNet.weights = [ w - np.multiply((eta/(np.sqrt(u+Optimizers.eps))),(Optimizers.beta1*m + ((1-Optimizers.beta1)/(1-Optimizers.beta1**Optimizers.t))*wG))  for w,m, u, wG in zip( neuralNet.weights,mWHat,uWHat, weightsGrad)]
        neuralNet.bias = [ b - np.multiply((eta/(np.sqrt(u+Optimizers.eps))),(Optimizers.beta1*m +  ((1-Optimizers.beta1)/(1-Optimizers.beta1**Optimizers.t))*bG))  for b,m, u, bG in zip( neuralNet.bias,mBHat,uBHat,biasGrad)]
        Optimizers.t += 1
    #init    
    def __init__(self):
        Optimizers.updateWeights = [0]*( NeuralNet.nHiddenLayers+1)
        Optimizers.updateBias = [0]*( NeuralNet.nHiddenLayers+1)
        Optimizers.mW = [0]*( NeuralNet.nHiddenLayers+1)
        Optimizers.mB = [0]*( NeuralNet.nHiddenLayers+1)
        Optimizers.uW = [0]*( NeuralNet.nHiddenLayers+1)
        Optimizers.uB = [0]*( NeuralNet.nHiddenLayers+1)




'''
sweep_config = {
    'name' : 'Working sweep',
    "method": "bayes",
    'metric': { 
        'name':'Loss',
        'goal': 'minimize',
        },
    'early_terminate':{
        'type': 'hyperband',
        'min_iter': 1
    },
    'parameters':{
        'learning_rate': {'values' : [.001]},
        'epochs' : {'values' : [10]},
        'optimizer':{'values' : ['nadam']},
        'n_hidden_layers' : {'values' : [4]},
        'size_hidden_layers' : {'values' : [64]},
        'batch_size' : {'values': [16]},
        'hidden_Layer_AF':{'values' : ['sigmoid']},
        'loss_func':{'values' : [ 'crossentropy']},
        'alpha' : {'values' : [0]},
        'initializer':{'values' : ['xavier']}
        }
}
'''

sweep_config = {
    'name' : 'Working sweep',
    "method": "random",
    'metric': { 
        'name':'Loss',
        'goal': 'minimize',
        },
    'early_terminate':{
        'type': 'hyperband',
        'min_iter': 1,
        'eta' : 1
    },
    'parameters':{
        'learning_rate': {'values' : [.001,0.0001]},
        'epochs' : {'values' : [10]},
        'optimizer':{'values' : ['sgd','nagd','mgd','rmsprop','adam','nadam']},
        'n_hidden_layers' : {'values' : [3,4,5]},
        'size_hidden_layers' : {'values' : [32,64,128]},
        'batch_size' : {'values': [16,32,64]},
        'hidden_Layer_AF':{'values' : ['tanh','sigmoid','relu']},
        'loss_func':{'values' : [ 'crossentropy']},
        'alpha' : {'values' : [0,0.0005,0.5]},
        'initializer':{'values' : ['random','xavier']}
        }
}

hyperparameter_defaults = dict(
    batch_size = 64,
    learning_rate = 0.001,
    epochs = 10,
    n_hidden_layers = 4,
    size_hidden_layers = 64,
    optimizer = "adam"
    )

def run():
    
    wandb.init(config=hyperparameter_defaults ,project="dl_assignment1",entity = "-my")
    config = wandb.config
    name = "hl_"+str(config.n_hidden_layers)+"_bs_"+str(config.batch_size) + "_sHL_" +str(config.size_hidden_layers) + "_ac_"+str(config.hidden_Layer_AF) + "_op_"+str(config.optimizer)
    wandb.init().name = name
    
    nHiddenLayers = config.n_hidden_layers
    neurons = []
    neurons.append(config.size_hidden_layers)
    neurons = neurons*nHiddenLayers
    eta = config.learning_rate
    epochs = config.epochs
    batchSize = config.batch_size
    optimizer = config.optimizer
    lossFunc = config.loss_func
    hLActivationFunc = config.hidden_Layer_AF
    alpha = config.alpha
    initializer = config.initializer
    '''
    #----------
    nHiddenLayers = 3
    neurons = []
    neurons.append(64)
    neurons = neurons*nHiddenLayers
    eta = 0.001
    epochs = 10
    batchSize = 16
    optimizer = "nadam"
    lossFunc = "mse"
    hLActivationFunc = "relu"
    alpha = 0.0
    initializer = 'random'
    '''
    #----------------

    outputNeurons = 10
    outputActivationFunc = "softmax"
    weights,bias = initializeNN(nHiddenLayers,neurons,outputNeurons,initializer)

    neuralNet = NeuralNet(nHiddenLayers,neurons,weights,bias,hLActivationFunc,outputActivationFunc,lossFunc)
    Optimizers()

    #NOTE : Function to send each class images to wandb (Commented so that it does not run everytime)
    #outEachClass()

    t = 1
    while t <= epochs:
        loss, accuracy = train(neuralNet, batchSize,optimizer,eta, alpha)  
        valLoss, valAccuracy, _ = predict(XVal,YVal,neuralNet)   
        print("Epoch : ",t,"Loss  = ",loss, " Accuracy : ", accuracy,"Val Loss  = ",valLoss, "Val Accuracy : ", valAccuracy)
        #wandb.log({'epoch':t,'Loss':loss, "Accuracy":accuracy,'Val_Loss':valLoss, "Val_Accuracy":valAccuracy})
        #wandb.log({"metric":loss })
        t+=1
    '''
    _, _, prediction = predict(x_train,y_train,neuralNet)
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            preds=prediction, y_true=np.reshape(y_train,(y_train.shape[0])).tolist(),
                            class_names=list(classLabels.values()))})
    '''
sweepId = wandb.sweep(sweep_config,entity = "-my",project = "dl_assignment1")
wandb.agent(sweepId,function=run)

#run()



