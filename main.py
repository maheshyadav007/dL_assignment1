import numpy as np
import wandb
from keras.datasets import fashion_mnist

wandb.init(project="dl_assignment1", entity="-my")
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

