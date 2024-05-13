#### IMPORTS ####
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.io as io
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys

#### MISC FUNCTIONS #####

# load MNIST training and test data
def getProcessedData():
    # get the data from the mnsit data
    train = datasets.MNIST(".", train=True, download=False)
    test = datasets.MNIST(".", train=False, download=False)

    # extract the image data into x and labels into y
    x_train = train.data.float()
    y_train = train.targets
    x_test = test.data.float()
    y_test = test.targets

    # convert the training and test data into numpy array
    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()

    # create validation data
    test_size = x_test.shape[0]
    indices = np.random.choice(x_train.shape[0], test_size, replace=False)

    x_valid = x_train[indices]
    y_valid = y_train[indices]

    # remove validation set from training set
    x_train = np.delete(x_train, indices, axis=0)
    y_train = np.delete(y_train, indices, axis=0)

    # flatten the image data into 1D arrays
    x_train = x_train.reshape(-1, 28*28)
    x_valid = x_valid.reshape(-1,28*28)
    x_test = x_test.reshape(-1, 28*28)

    return x_train, y_train, x_valid, y_valid,  x_test, y_test

def makeDatasets(x_train, y_train, x_valid, y_valid,  x_test, y_test):
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return train_dataset, val_dataset, test_dataset

def make_DataLoaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    return train_loader, val_loader, test_loader

# read JPEG from path and convert to flatted tensor
def JPGtoTensor(file_path,device):
    image = io.read_image(file_path)
    image = image.float()
    image_np = image.numpy()
    image = torch.tensor(image_np).view(1, -1)
    image = image.to(device)
    return image

def display_image(X_i, title):
    plt.imshow(X_i, cmap='binary')
    plt.title(title)
    plt.show()

#### ANN MODEL ####

class NumberClassifier(nn.Module):
    def __init__(self, input_size=28*28, num_layers=3, layer_sizes=[28*28,28*28,28*28], output_size=10, activation=F.relu, dropout_rate = 0):
        super(NumberClassifier, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Add hidden layers
        for i in range(num_layers):
            self.layers.append(nn.Linear(input_size, layer_sizes[i]))
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(p=dropout_rate))
            input_size = layer_sizes[i]

        # create output and activation layers
        self.output_layer = nn.Linear(layer_sizes[len(layer_sizes)-1], output_size)
        self.activation = activation

    def forward(self, feature):
        feature = feature.float()
        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                feature = layer(feature)
            else:
                feature = self.activation(layer(feature))
        feature = torch.softmax(self.output_layer(feature), dim=1)
        return feature

def train_classifier(model,train_loader, val_loader,max_no_epochs, min_loss_chng,learn_rate, display=False, loss_func=nn.CrossEntropyLoss()):
    # initialize previous loss function and
    optimizer = torch.optim.SGD(model.parameters(),lr=learn_rate)

    # initialize trends and previous loss
    validation_trend = []
    training_trend = []
    loss_trend = []
    epochs = []
    no_min_change = 0
    no_saved = 0
    for epoch in range(0,max_no_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)  # Forward pass
            loss = loss_func(output, target)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the parameters

            if (batch_idx+1) % 100 == 0:
                
                val_acc = validate_classifier(model,val_loader)
                train_acc = validate_classifier(model,train_loader, max_batches=len(val_loader))

                validation_trend.append(val_acc)
                training_trend.append(train_acc)
                loss_trend.append(loss.item())
                epochs.append(epoch + ((batch_idx + 1) / len(train_loader)))

                if display: print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch + 1, max_no_epochs, batch_idx+1, len(train_loader), loss.item()) + f" Validation Accuracy:{val_acc:.2f}%" + f" Training Accuracy:{train_acc:.2f}%")

                no_saved += 1

        # valculate the accuracy change to determine if model has converged
        if no_saved > 1: # check that previous loss has been set
            loss_chng = abs(loss_trend[no_saved - 2] - loss_trend[no_saved - 1])/loss_trend[no_saved - 1]
            if loss_chng < min_loss_chng: # stop trainning if loss is not changing
                no_min_change += 1
            else:
                no_min_change = 0
            if no_min_change > 4:
                return validation_trend, training_trend, loss_trend, epochs
            
    return validation_trend, training_trend, loss_trend, epochs

def validate_classifier(model, test_loader, max_batches = None):
    model.eval()  # set model to evaluation mode
    correct = 0
    total = 0
    num_batches = 0
    with torch.no_grad():  # disable gradient calculation for validation
        for data, target in test_loader:
            output = model(data)  # forward pass
            _, predicted = torch.max(output.data, 1)  # get predicted labels
            total += target.size(0)
            correct += (predicted == target).sum().item()  # count correct predictions
            num_batches += 1

            if max_batches is not None and num_batches >= max_batches:
                break  # stop processing batches if max_batches is reached

    accuracy = 100 * correct / total

    return accuracy

def makePrediction(model,image):
    output = model(torch.tensor(image).view(1, -1))
    _, predicted = torch.max(output, 1)
    return predicted

def validate_jpgs(model, device, dir_path):
    correct = 0
    total = 0
    numbers = [0,1,2,3,4,5,6,7,8,9]

    # loop through every letter/class in alphabet
    for class_num in numbers:
        # set the class directory path
        class_dir_pth = dir_path + "/" + str(class_num)

        # get all the jpg files in the class directory
        files = os.listdir(class_dir_pth)
        jpg_files = [file for file in files if file.lower().endswith('.jpg')]

        for file in jpg_files:
            image = JPGtoTensor(class_dir_pth + "/" + file,device=device)
            predicted = makePrediction(model,image)
            total += 1
            if predicted == class_num:
                correct += 1

    accuracy = 100 * correct / total
    return accuracy

# save the current state of a model
def save_model(model,file_path):
    torch.save(model.state_dict(), file_path)

# load a previously trained model
def load_model(file_path):
    model = NumberClassifier()
    model.load_state_dict(torch.load(file_path))
    return model