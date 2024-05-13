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
def getKaggleData(file_path, test_size=0.2, test = True):
    # Read the data from the file path
    data = pd.read_csv(file_path)

    # Extract the image data into X and labels into y
    y = data['0']
    X = data.drop('0', axis=1)
    X = X.astype(float)

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_size, random_state=42)

    # Create validation data
    if test:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
    else:
        X_valid, y_valid = None, None

    # Reshape the image data into 2D arrays
    X_train = X_train.reshape(-1, 28 * 28)
    X_valid = X_valid.reshape(-1, 28 * 28) if X_valid is not None else None
    X_test = X_test.reshape(-1, 28 * 28)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

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

class LetterClassifier(nn.Module):
    def __init__(self, input_size=28*28, num_layers=3, layer_sizes=[28*28,28*28], output_size=10, activation=F.relu, dropout_rate = 0):
        super(LetterClassifier, self).__init__()
        
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

def examplePredictions(model,no_examples):
    model.eval()
    with torch.no_grad():
        for i in range(no_examples):
            idx = random.randint(0, x_test.shape[0])
            image = x_test[idx]
            label = y_test[idx]
            predicted = makePrediction(model,image)
            print("True label: %d, Predicted label: %d" % (label, predicted))
            display_image(image.reshape(28, 28), "True label: %d, Predicted label: %d" % (label, predicted))

# save the current state of a model
def save_model(model,file_path):
    torch.save(model.state_dict(), file_path)

# load a previously trained model
def load_model(file_path):
    model = LetterClassifier()
    model.load_state_dict(torch.load(file_path))
    return model

#### MAIN ####
if __name__ == "__main__":

    # set constant training values
    no_epochs = 6
    min_loss_chng = 0.02
    learn_rate = 0.01
    batch_size = 32

    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the dataset
    print("Loading Dataset.")
    x_train, y_train, x_valid, y_valid,  x_test, y_test = getKaggleData('data/kaggle/a_to_j.csv')
    train_dataset,val_dataset,test_dataset = makeDatasets(x_train,y_train,x_valid, y_valid, x_test,y_test)

    train_dataset = [(x.to(device), y.to(device)) for x, y in train_dataset]
    val_dataset = [(x.to(device), y.to(device)) for x, y in val_dataset]
    test_dataset = [(x.to(device), y.to(device)) for x, y in test_dataset]

    train_loader, val_loader, test_loader = make_DataLoaders(train_dataset,val_dataset,test_dataset,batch_size)
    print("Data Loaded Successfully.")

    # load last model if exists and user requests it
    loaded = False
    if os.path.exists('current_model.pth'):
        load_new = input("Would you like to load the previous model (y/n):")
        if load_new == "y":
            # File path exists, proceed with loading the model
            classifier = LetterClassifier(input_size= 28*28, num_layers=3,layer_sizes=[28*28,28*28,28*28],output_size=13)
            classifier.load_state_dict(torch.load('current_model.pth'))
            classifier.to(device)
            print("Model loaded successfully.")
            loaded = True
        
    # train the model if not loaded
    if loaded == False:
        print("Beginning training.")
        print("Training using:", device)
        classifier = LetterClassifier(input_size= 28*28, num_layers=3,layer_sizes=[28*28,28*28,28*28],output_size=13)
        classifier.to(device)
        validation_trend, training_trend, loss_trend, epochs = train_classifier(classifier,train_loader,val_loader,no_epochs,min_loss_chng,learn_rate, display = True)

    # show the models test accuracy
    val_acc = validate_classifier(classifier, test_loader)
    print(f"Test Set Validation Accuracy: {val_acc:.2f}%")

    # ask if teh user would like to save the current model
    save = input("Would you like to overwrite the last model (y/n):")
    if save == "y":
        save_model(classifier,'current_model.pth')
        print("Model saved successfully.")

    # get file path of digit to predict
    file_path = input("Please enter a filepath (\"q\" to quit): ")
    while file_path != "q":
        image = JPGtoTensor(file_path,device)
        prediction = makePrediction(classifier,image)
        print("Classifier: %d" % prediction)
        file_path = input("Please enter a filepath (\"q\" to quit): ")