import matplotlib.gridspec as gridspec
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import torch
import sys
import string
import scripts.building_abandoned as building_abandoned

# [validation_trend, training_trend, loss_trend, epochs]
def save_trends(train_results,test_acc,n_l,acc_type,dir_save_path = "results/trends/test_"):
    for i in len(train_results):
        # ensure both arrays have the same length
        if len(train_results[i][0]) != len(train_results[i][1]):
            raise ValueError("Arrays must have the same length")

        # zip the arrays together
        data = zip(train_results[i][0], train_results[i][1], train_results[i][2], train_results[i][3])

        # create file_name and path
        file_path = dir_save_path
        file_path += str(n_l) + "_"
        file_path += str(acc_type.__name__.replace("_", "")) + "_"
        file_path += str(test_acc).replace('.', '_') 
        file_path += "_" + str(i) + ".csv"

        # write the data to a CSV file
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['validation_tren', 'training_trend', 'loss_trend', 'epochs']) 
            writer.writerows(data)

def read_trends_and_plot(csv_file,dir_path="results/plots/"):
    # extract test parameters from the file name
    file_name = csv_file.split("/")[-1].replace('.csv', '')  # extract only the file name and remove extension
    params = file_name.split("_")
    no_layers = int(params[1])
    activation_type = params[2]
    test_acc = float(params[3] + "." + params[4])

    # read data from CSV
    val_trend = []
    train_trend = []
    loss_trend = []
    epochs = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            val_trend.append(float(row[0]))
            train_trend.append(float(row[1]))
            loss_trend.append(float(row[2]))
            epochs.append(float(row[3]))

    # plot trends
    plt.figure(figsize=(10, 10))

    # define gridspec
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # top subplot for validation and training trends
    ax1 = plt.subplot(gs[0])
    ax1.plot(epochs, val_trend, label='Validation Trend')
    ax1.plot(epochs, train_trend, label='Training Trend')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Validation and Training Trends\nLayers: {no_layers}, Activation: {activation_type}, Learning Rate: {learning_rate}, Test Accuracy: {test_acc}')
    ax1.legend()

    # bottom subplot for loss trend
    ax2 = plt.subplot(gs[1])
    ax2.plot(epochs, loss_trend, label='Loss Trend', color='green')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Trend')
    ax2.legend()

    # adjust layout
    plt.tight_layout()

    # save plot as a figure
    plot_file_name = dir_path + os.path.basename(csv_file)[:-4] + ".png"
    plt.savefig(plot_file_name)
    plt.close()
   
def traverse_and_plot(directory,dir_save_path="results/plots/"):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_file = os.path.join(root, file)
                read_trends_and_plot(csv_file,dir_path=dir_save_path)

if __name__ == "__main__":
    no_epochs = 1
    min_loss_chng = 0.02
    batch_size = 32

    # check if user wants to keep old tests
    keep = input("Would you like to delete previous test? (y/n):")
    if keep == "y":
        sure = input("Are you sure ? (y/n):")
        if sure == "y":
            for root, dirs, files in os.walk("results"):
                # Iterate over each file and delete it
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
    print("Deleted old results.")

    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set tha test parameters
    no_layers = [1,2,3]
    activation_types = [F.relu,F.leaky_relu,F.elu]
    l_r = 0.1

    print("Loaded Data.")

    # initialise pandas dataframe
    df = pd.DataFrame(columns=['activation_function', 'no_layers', 'layer_sizes', 'learning_rate', 'test_accuracy'])

    print("Starting training process.")


    # testing single model
    for n_l in no_layers:
        for acc_type in activation_types:
            # create and train model
            comb_classifier = building_abandoned.CombLetterClassifier(num_models = 1, activation=acc_type,device=device,model_targets=[])

            # loading data
            print("Loading Data.")
            comb_classifier.createDataLoaders("data/kaggle/kaggle_letters.csv",batch_size)
            print("Data Loaded.")

            # train the model
            train_results = comb_classifier.train(no_epochs,min_loss_chng,l_r,display=True)

            # print succesfule training
            print("Trained with " + acc_type.__name__ + " with " + str(n_l) + " layers at learning rate of " + str(l_r))

            # validate model
            accuracy = comb_classifier.validate("data/hand_crafted")
            print("Combined Model Test Accuracy: {:.4f}".format(accuracy))

            # save trends to a csv
            save_trends(train_results,accuracy,n_l,acc_type)

            # add data to the pandas df
            record = {
                'activation_function': acc_type.__name__,
                'no_layers': n_l,
                'test_accuracy': accuracy
            }
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

            torch.cuda.empty_cache()
            print("GPU memory released.")

    traverse_and_plot("results/trends")
    df.to_csv("results/test_accuracies.csv", index=False)