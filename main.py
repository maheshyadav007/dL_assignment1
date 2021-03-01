import numpy as np
import wandb
from keras.datasets import fashion_mnist

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
    wandb.log({"Categories": [wandb.Image(image, caption=caption) for image, caption in zip(data,labels)]})

n,l,w = x_train.shape
nFeatures = l*w
X = np.reshape(x_train,(n,l*w,1))
Y = np.reshape(y_train,(n,1))

def initializeParams(nFeatures,weights,bias,nHiddenLayers,neurons):
    n = nFeatures
    for i in range(nHiddenLayers+1):
        weights.append(np.random.rand(n,neurons[i]))
        bias.append(np.random.rand(neurons[i],1))
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

nHiddenLayers = 3
neurons = [8, 6, 7]
outputNeurons = 10
weights, bias = initializeNN(nHiddenLayers,neurons,outputNeurons)

#print(weights)
#print(bias)
def preActivate(x,w,b):
    #print(w.shape, x.shape)
    #print(len(x),w)
    #print(x.shape,w.shape,b.shape)
    out = np.dot(np.transpose(w),x)+b
    return out
def logisticActivation(parameter):
    out = (1/(1+np.exp(-1*parameter)))
    return out
def softmaxActivation(parameter):
    exp = np.exp(parameter)
    out = exp/np.sum(exp)
    return out
def feedForwardNN(weights,bias):
    nHiddenLayers = len(bias)-1
    x = X[0]
    a = []
    h = []
    for i in range(nHiddenLayers):
        pA = preActivate(x,weights[i],bias[i])
        activation = logisticActivation(pA)
        x = activation
        a.append(pA)
        h.append(activation)
    pA = preActivate(x,weights[nHiddenLayers],bias[nHiddenLayers])
    a.append(pA)
    output = softmaxActivation(pA)
    return a,h,output

a,h,yHat = feedForwardNN(weights,bias)
print(a[0].shape, h[0].shape, yHat.shape)

def computeLoss():
    loss = 0
    return loss

def derivativeActivation(activation,parameter):
    a = 0
    if activation == "logistic":
        a = logisticActivation(parameter)
        a = np.multiply(a,(1-a))
    return a

def backPropagation(a,h,y,yHat,weights):
    weightsGrad = []
    biasGrad = []
    el = np.zeros(shape = (10,1))
    el[y] = 1
    aLGrad = -(el-yHat)
    #check dimensions of alGrad
    #print(aLGrad.shape)
    for k in range(nHiddenLayers,0,-1):
        weightsGrad.append(np.dot(aLGrad,np.transpose(h[k-1])))
        biasGrad.append(aLGrad)
        #watch for transpose
        #a = np.dot(np.transpose(weights[k]),aLGrad)
        a = np.dot((weights[k]),aLGrad)
        aLGrad = np.multiply(a,derivativeActivation("logistic",a[k-1]))

backPropagation(a,h,Y[0],yHat,weights)