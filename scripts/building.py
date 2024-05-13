#### IMPORTS ####
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.io as io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os
import sys
import string

################## MISC FUNCTIONS ###################

# split dataset into training validation and testing
def getSplitData(data, test_size=0.2, test = True):

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

# make datasets 
def makeDatasets(x_train, y_train, x_valid, y_valid,  x_test, y_test):
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return train_dataset, val_dataset, test_dataset

# make data loaders
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

# display image
def display_image(X_i, title):
    plt.imshow(X_i, cmap='binary')
    plt.title(title)
    plt.show()

################## ANN MODEL ###################

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

# used to train a single letter classifier model
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
    
    del optimizer
    del loss_func
    return validation_trend, training_trend, loss_trend, epochs

# used to classify a single letter classifier
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

# make prediction using LetterClassifier
def makePrediction(model,image):
    output = model(torch.tensor(image).view(1, -1))
    _, predicted = torch.max(output, 1)
    return predicted

# save the current state of a model
def save_model(model,file_path):
    torch.save(model.state_dict(), file_path)

# load a previously trained model
def load_model(file_path):
    model = LetterClassifier()
    model.load_state_dict(torch.load(file_path))
    return model

############### COMBINED LETTER CLASSIFIER ANNS ###############
class CombLetterClassifier:
    # initialize
    def __init__(self, num_models, device,input_size=28*28, num_layers=3, layer_sizes=[28*28,28*28,28*28], output_size=26, activation=F.relu, dropout_rate = 0, create_models=True,model_targets = []):
        # initlialize standard variables
        self.num_models = num_models
        self.models = []
        self.device = device
        self.model_targets = model_targets
        self.output_size = output_size

        # split output classes among models
        if len(self.model_targets) == 0 or len(self.model_targets) != self.num_models:
            print("Using default target split")
            quotient, remainder = divmod(output_size, num_models)
            self.output_sizes =  [quotient + 1] * remainder + [quotient] * (num_models - remainder)
        else:
            self.output_sizes = []
            for i in range(len(model_targets)):
                self.output_sizes.append(len(model_targets[i]))

        # create the relevant number of models
        if create_models:
            i = 0
            for o_s in self.output_sizes:
                self.models.append(LetterClassifier(input_size=input_size, num_layers=num_layers, layer_sizes=layer_sizes, output_size=o_s, activation=activation, dropout_rate = dropout_rate))
                self.models[i].to(self.device)
                i=i+1

    # create the data loders for each model
    def createDataLoaders(self,data_file_path,batch_size):
        # intialise dataloader arrays
        self.train_loaders = []
        self.test_loaders = []
        self.val_loaders = []

        # load the entire dataset
        data = pd.read_csv(data_file_path)

        start = 0
        for i in range(self.num_models):   # for loop where each iteration creates new set of dataloaders per model
            print("Making loaders for model ",i, "with num_classes = ",self.output_sizes[i])
            print("start =",start)

            # calculate class intervals
            end = start + self.output_sizes[i]

            # reduce dataset to specific classes
            downsampled_data = []
            if len(self.model_targets) == 0 or len(self.model_targets) != self.num_models:
                for j in range(start,end):
                    class_data = data[data.iloc[:, 0] == j]
                    class_data[data.columns[0]] = class_data[data.columns[0]] - start
                    downsampled_data.append(class_data)
                downsampled_data = pd.concat(downsampled_data)
            else:
                for j in range(0,self.output_size):
                    if j in self.model_targets[i]:
                        class_data = data[data.iloc[:, 0] == j]
                        class_data[data.columns[0]] = self.model_targets[i].index(j)
                        downsampled_data.append(class_data)
                downsampled_data = pd.concat(downsampled_data)

            # split downsampled data into train val and test
            X_train, y_train, X_valid, y_valid, X_test, y_test = getSplitData(downsampled_data, test_size=0.2, test = True)

            # make datasets and move to device
            train_dataset,val_dataset,test_dataset = makeDatasets(X_train,y_train,X_valid, y_valid, X_test,y_test)
            train_dataset = [(x.to(self.device), y.to(self.device)) for x, y in train_dataset]
            val_dataset = [(x.to(self.device), y.to(self.device)) for x, y in val_dataset]
            test_dataset = [(x.to(self.device), y.to(self.device)) for x, y in test_dataset]
            print(np.unique(y_train))

            # make dataloaders and add to arrays
            train_loader, val_loader, test_loader = make_DataLoaders(train_dataset,val_dataset,test_dataset,batch_size)
            self.train_loaders.append(train_loader)
            self.val_loaders.append(val_loader)
            self.test_loaders.append(test_loader)

            start = end

    # train all the models on thier relevant training data
    def train(self,max_no_epochs, min_loss_chng,learn_rate, display=False, loss_func=nn.CrossEntropyLoss()):
        for i in range(self.num_models):
            print("Training model:",i)
            train_classifier(self.models[i],self.train_loaders[i],self.val_loaders[i],max_no_epochs,min_loss_chng,learn_rate,display=display,loss_func=loss_func)
            print("Validating model:",i)
            test_acc = validate_classifier(self.models[i],self.test_loaders[i])
            print(f"Test Set Validation Accuracy: {test_acc:.2f}%")
    
    # classify single image
    def classify(self,image):

        # get the predictions from each model
        max_preds = []
        for i in range(self.num_models):
            output = self.models[i](image.clone().detach().view(1, -1))
            max_value, predicted = torch.max(output, 1) # find the maximum prediction
            if len(self.model_targets) == 0:
                max_preds.append([max_value.item(),predicted.item() + sum(self.output_sizes[:i])])  # add the maximum prediction and the corrected class
            else:
                max_preds.append([max_value.item(),self.model_targets[i][predicted.item()]])  # add the maximum prediction and the corrected class
        
        # find the most confident prediction
        max = None
        for max_pred in max_preds:
            if max == None:
                max = max_pred
            else:
                if max_pred[0] > max[0]:
                    max = max_pred

        return max[1] # return the predicted class

    def validate(self, dir_path):
        correct = 0
        total = 0
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # loop through every letter/class in alphabet
        for class_num, letter in enumerate(alphabet):
            # set the class directory path
            class_dir_pth = dir_path + "/" + letter

            # get all the jpg files in the class directory
            files = os.listdir(class_dir_pth)
            jpg_files = [file for file in files if file.lower().endswith('.jpg')]

            for file in jpg_files:
                image = JPGtoTensor(class_dir_pth + "/" + file,device=self.device)
                predicted = self.classify(image)
                total += 1
                if predicted == class_num:
                    correct += 1
                else:
                    alphabet = string.ascii_uppercase
                    print("Predicted: " + alphabet[predicted] + "\tActual: " + alphabet[class_num])

        accuracy = 100 * correct / total

        return accuracy

    # save all models to standard file location
    def save(self):
        for i in range(self.num_models):
            save_model(self.models[i],"models/model_" + str(i) + ".pth")
    
    # load previously saved models
    def load(self,num_models, device,input_size=28*28, num_layers=3, layer_sizes=[28*28,28*28,28*28], activation=F.relu, dropout_rate = 0):
        self.num_models = num_models
        self.models = []
        for i in range(self.num_models):
            self.models.append(LetterClassifier(input_size=input_size, num_layers=num_layers, layer_sizes=layer_sizes, output_size=self.output_sizes[i], activation=activation, dropout_rate = dropout_rate))
            self.models[i].load_state_dict(torch.load("models/model_" + str(i) + ".pth"))
            self.models[i].to(device)


#### MAIN ####
if __name__ == "__main__":
    try:
        # set constant training values
        no_epochs = 30
        min_loss_chng = 0.02
        learn_rate = 0.01
        batch_size = 32
        num_models = 1

        # target model list
        alphabet = string.ascii_uppercase
        # letters_list = [['D', 'O', 'G', 'Q', 'U', 'V', 'W', 'Y', 'Z', 'M', 'N', 'F', 'T'],
        #         ['E', 'I', 'H', 'L', 'K', 'A', 'R', 'X', 'C', 'J', 'P', 'B', 'S']]
        # letters_list = [['B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'P', 'R', 'S'], ['A', 'I', 'M', 'N', 'O', 'Q', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']]
        # letters_list = [['M', 'N', 'W', 'H', 'V', 'U', 'R', 'P', 'Y'],
        #                 ['O', 'C', 'D', 'Q', 'G', 'B', 'A', 'K'],
        #                 ['E', 'F', 'X', 'T', 'I', 'L', 'J', 'S', 'Z']]
        letters_list = []
        positions_list = [[alphabet.index(letter) for letter in sublist] for sublist in letters_list]

        # get the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialise model
        comb_classifier = CombLetterClassifier(num_models,device,create_models=False,model_targets=positions_list)

        # load last model if exists and user requests it
        loaded = False
        load_new = input("Would you like to load the previous model (y/n):")
        if load_new == "y":
            # File path exists, proceed with loading the model
            comb_classifier.load(num_models,device)
            print("Model loaded successfully.")
            loaded = True
            
        # train the model if not loaded
        if loaded == False:

            comb_classifier = CombLetterClassifier(num_models,device,model_targets=positions_list)

            print("Loading Data.")
            comb_classifier.createDataLoaders("data/kaggle/kaggle_letters.csv",batch_size)
            print("Data Loaded.")

            print("Training Models.")
            comb_classifier.train(no_epochs,min_loss_chng,learn_rate,display=True)

            # ask if teh user would like to save the current model
            save = input("Would you like to overwrite the last model (y/n):")
            if save == "y":
                comb_classifier.save()
                print("Model saved successfully.")

        # test final model on handcrafted test data
        accuracy = comb_classifier.validate("data/hand_crafted")
        print("Combined Model Test Accuracy: {:.4f}".format(accuracy))

        # get file path of digit to predict
        file_path = input("Please enter a filepath (\"q\" to quit): ")
        while file_path != "q":
            if file_path != "":
                image = JPGtoTensor(file_path,device)
                pred_class = comb_classifier.classify(image)
                print("Im pretty sure its",alphabet[pred_class])
            file_path = input("Please enter a filepath (\"q\" to quit): ")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")
        torch.cuda.empty_cache()
        print("GPU memory released.")
        sys.exit(1)