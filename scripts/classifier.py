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
import building

#### MAIN ####
if __name__ == "__main__":
    try:
        # set constant training values
        no_epochs = 10
        min_loss_chng = 0.02
        learn_rate = 0.01
        batch_size = 32
        num_layers = 3

        # get the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load the dataset
        print("Loading Dataset.")
        x_train, y_train, x_valid, y_valid,  x_test, y_test = building.getProcessedData()
        train_dataset,val_dataset,test_dataset = building.makeDatasets(x_train,y_train,x_valid, y_valid, x_test,y_test)

        train_dataset = [(x.to(device), y.to(device)) for x, y in train_dataset]
        val_dataset = [(x.to(device), y.to(device)) for x, y in val_dataset]
        test_dataset = [(x.to(device), y.to(device)) for x, y in test_dataset]

        train_loader, val_loader, test_loader = building.make_DataLoaders(train_dataset,val_dataset,test_dataset,batch_size)
        print("Data Loaded Successfully.")

        # load last model if exists and user requests it
        loaded = False
        if os.path.exists('current_model.pth'):
            load_new = input("Would you like to load the previous model (y/n):")
            if load_new == "y":
                # File path exists, proceed with loading the model
                classifier = building.NumberClassifier(input_size= 28*28, num_layers=num_layers,output_size=10)
                classifier.load_state_dict(torch.load('current_model.pth'))
                classifier.to(device)
                print("Model loaded successfully.")
                loaded = True
            
        # train the model if not loaded
        if loaded == False:
            print("Beginning training.")
            print("Training using:", device)
            classifier = building.NumberClassifier(input_size= 28*28, num_layers=num_layers,output_size=10)
            classifier.to(device)
            validation_trend, training_trend, loss_trend, epochs = building.train_classifier(classifier,train_loader,val_loader,no_epochs,min_loss_chng,learn_rate, display = True)

        # show the models test accuracy
        # val_acc = building.validate_classifier(classifier, test_loader)
        val_acc = building.validate_jpgs(classifier,device,'data/hand_crafted')
        print(f"Test Set Validation Accuracy: {val_acc:.2f}%")

        # ask if teh user would like to save the current model
        if loaded == False:
            save = input("Would you like to overwrite the last model (y/n):")
            if save == "y":
                building.save_model(classifier,'current_model.pth')
                print("Model saved successfully.")

        # get file path of digit to predict
        file_path = input("Please enter a filepath (\"q\" to quit): ")
        while file_path != "q":
            image = building.JPGtoTensor(file_path,device)
            prediction = building.makePrediction(classifier,image)
            print("Classifier: %d" % prediction)
            file_path = input("Please enter a filepath (\"q\" to quit): ")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")
        torch.cuda.empty_cache()
        print("GPU memory released.")
        sys.exit(1)