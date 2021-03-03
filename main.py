import numpy as np
import wandb
from keras.datasets import fashion_mnist


#NOTE
#1/N in loss function



#wandb.init(project="dl_assignment1", entity="-my")



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

n,l,w = x_train.shape
nFeatures = l*w
X = np.reshape(x_train,(n,l*w,1))/255
Y = np.reshape(y_train,(n,1))

def initializeParams(nFeatures,weights,bias,nHiddenLayers,neurons):
    n = nFeatures
    for i in range(nHiddenLayers+1):
        weights.append(np.random.randint(-1,1,(neurons[i],n)))
        bias.append(np.random.randint(-1,1,(neurons[i],1)))
        n = neurons[i]
def initializeNN(nHiddenLayers,neurons,outputNeurons): 
    weights = []
    bias = []
    #nHiddenLayers = 2
    #neurons = [2, 3]
    #outputNeurons = 10
    neurons.append(outputNeurons)
    initializeParams(nFeatures,weights,bias,nHiddenLayers,neurons)

    return weights, bias




#print(weights)
#print(bias)
def preActivate(x,w,b):
    
    #out = np.dot(np.transpose(w),x)+b
    out = np.dot((w),x)+b                       #transpose removed
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
        #print(k)
        weightsGrad[k] = (np.dot(aLGrad,np.transpose(h[k-1])))
        biasGrad[k] = (aLGrad)
        #watch for transpose
        #a = np.dot(np.transpose(weights[k]),aLGrad)
        a = np.dot(np.transpose(weights[k]),aLGrad)                         #transpose added
        aLGrad = np.multiply(a,derivativeActivation("logistic",a[k-1]))
    #print(aLGrad,x)
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
            #x, y = X[0], Y[0]
            a,h,yHat = feedForwardNN(x,weights,bias)
            #print(a[0].shape, h[0].shape, yHat.shape)
            loss += computeLoss(yHat,y)
            wGrad, bGrad = backPropagation(a,h,x,y,yHat,weights)
            if t==0:
                weightsGrad = wGrad
                biasGrad = bGrad
            else :
                #print("in lese")
                weightsGrad += np.array(wGrad)
                biasGrad += np.array(bGrad)
            
        print(loss)
        wandb.log({'loss':loss})
        #print(weights, weightsGrad)
        weights = weights - np.multiply(eta,weightsGrad)
        bias = bias - np.multiply(eta,biasGrad)

        t+=1


def train(weights,bias,x,y,optimizer,lossFunction,nHiddenLayers):
    a,h,yHat = feedForwardNN(x,weights,bias)
    
    loss = 0
    if lossFunction == "SoftMax":
        loss += computeSoftMaxLoss(yHat,y)
    
    wGrad, bGrad = backPropagation(a,h,x,y,yHat,weights,nHiddenLayers)# check y bias is not used

    return loss, wGrad, bGrad , yHat

'''
sweep_config = {
    'name' : 'MY sweep',
    "method": "random",
    'metric': { 
        'name':'loss',
        'goal': 'minimize',
        },
    'parameters':{
        'learning_rate': {'values' : [.001, 0.0001]},
        'epochs' : {'values' : [5,10]},
        'optimizer':{'values' : ['sgd']},
        'n_hidden_layers' : {'values' : [3,4,5]},
        'size_hidden_layers' : {'values' : [32,64,128]},
        'batch_size' : {'values': [1,16,32,64]}
        }
}'''
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
hyperparameter_defaults = dict(
    batch_size = 64,
    learning_rate = 0.001,
    epochs = 5,
    n_hidden_layers = 3,
    size_hidden_layers = 32,
    optimizer = "sgd"
    )


def validate(yHat, y, count):
    if np.argmax(yHat) == y[0]:
        return True
    return False

def run():
    
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
    '''
    #----------
    #config = wandb.config
    nHiddenLayers = 4
    neurons = []
    neurons.append(16)
    neurons = neurons*nHiddenLayers
    eta = 0.001
    epochs = 100
    batchSize = 64
    optimizer = "sgd"
    '''
    #----------------
    outputNeurons = 10
    lossFunction = "SoftMax"

    weights, bias = initializeNN(nHiddenLayers,neurons,outputNeurons)

    
    t = 1
    
    count = 0
    weightsGrad = [0]*(nHiddenLayers+1)
    biasGrad = [0]*(nHiddenLayers+1)
    while t <= epochs:
        count = 0
        Xb,Yb = X,Y
        loss = 0
        startIndex = 0
        notDone = True
       
        flag = True
        while notDone:
            if startIndex+batchSize <= n :
                Xb = X[startIndex:startIndex+batchSize,:]
                Yb = Y[startIndex:startIndex+batchSize, :]
            else :
                Xb = X[startIndex:-1,:]
                Yb = Y[startIndex:-1,:]
                notDone = False

            startIndex = startIndex+batchSize
            
            
            
            for x,y in zip(Xb,Yb):
                l, wGrad, bGrad, yHat = train(weights,bias,x,y,optimizer,lossFunction,nHiddenLayers)
                if validate(yHat,y, count) :
                    count += 1
                if flag:
                    weightsGrad = wGrad
                    biasGrad = bGrad
                    weightsGrad[0] /= batchSize
                    biasGrad[0] /= batchSize
                    flag = False
                else :
                    #print("in lese")
                    #weightsGrad += np.array(wGrad)
                    #biasGrad += np.array(bGrad)
                    weightsGrad = [ w + wG/batchSize  for w, wG in zip(weightsGrad, wGrad)]
                    biasGrad = [ b + bG/batchSize  for b, bG in zip(biasGrad, bGrad)]
                #weightsGrad += np.array(wGrad)
                #biasGrad += np.array(bGrad)
                loss += l
            #weights = weights - [wGrad * eta for wGrad in weightsGrad]
            #bias = bias - [bGrad * eta for bGrad in biasGrad]
            weights = [ w - eta*wG  for w, wG in zip(weights,weightsGrad)]
            bias = [ b - eta*bG  for b, bG in zip(bias,biasGrad)]
            
                
        print("After epoch : ",t,"Loss  = ",loss/60000, " Accuracy : ", count/60000)
        wandb.log({'epochs':epochs,'loss':loss})
        wandb.log({"metric":loss })
        t+=1


sweepId = wandb.sweep(sweep_config,entity = "-my",project = "dl_assignment1")
wandb.agent(sweepId,function=run)

#run()