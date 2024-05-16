import pandas as pd
import numpy as np
from PIL import Image
import os
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

import os
from PIL import Image

def getProcessedData(save_path):
    # get the data from the MNIST dataset
    train = datasets.MNIST(".", train=True, download=False)
    test = datasets.MNIST(".", train=False, download=False)

    # extract the image data into x and labels into y
    x_train = train.data.float()
    y_train = train.targets
    x_test = test.data.float()
    y_test = test.targets

    # convert the training and test data into numpy arrays
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
    x_train = x_train.reshape(-1, 28, 28)
    x_valid = x_valid.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)

    # Create directories for each class and save images
    for class_label in range(10):  # Assuming there are 10 classes (digits 0-9)
        class_dir = os.path.join(save_path, str(class_label))
        os.makedirs(class_dir, exist_ok=True)  # Create directory if it doesn't exist

        # Find indices of images corresponding to the current class
        class_indices = np.where(y_train == class_label)[0][:115]  # Take 115 images for each class

        # Save images as JPEG files
        for idx, image_idx in enumerate(class_indices):
            image = Image.fromarray(x_train[image_idx])
            image = image.convert('L')  # Convert image to grayscale
            image_path = os.path.join(class_dir, f"{class_label}_{idx}.jpg")
            image.save(image_path)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

# Usage example:
save_path = "data/MNIST_FAKE_TEST"
getProcessedData(save_path)
