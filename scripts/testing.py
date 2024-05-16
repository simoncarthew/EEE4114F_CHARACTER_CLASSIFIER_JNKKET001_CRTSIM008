import building
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import torch
import sys
import subprocess

def save_trends(model,val_trend, train_trend, loss_trend, epochs, test_acc, no_layers, activation_type, learning_rate,dir_save_path = "results/trends/test_"):
    # ensure both arrays have the same length
    if len(val_trend) != len(train_trend):
        raise ValueError("Arrays must have the same length")

    # zip the arrays together
    data = zip(val_trend, train_trend, loss_trend, epochs)

    # create file_name and path
    file_path = dir_save_path
    file_path += str(no_layers) + "_"
    file_path += str(activation_type.__name__.replace("_", "")) + "_"
    file_path += str(learning_rate)[2:] + "_"
    file_path += str(round(test_acc,2)).replace('.', '_') + ".csv"

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
    learning_rate = float("0." + params[3])
    test_acc = float(params[4] + "." + params[5])

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
    plt.figure(figsize=(8, 8))
    line_width = 2

    # define gridspec
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # top subplot for validation and training trends
    ax1 = plt.subplot(gs[0])
    ax1.plot(epochs, val_trend, label='Validation Trend', linewidth = line_width)
    ax1.plot(epochs, train_trend, label='Training Trend', linewidth = line_width)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    # ax1.set_title(f'Validation and Training Trends\nLayers: {no_layers}, Activation: {activation_type}, Learning Rate: {learning_rate}, Test Accuracy: {test_acc}')
    ax1.set_title(f'Layers: {no_layers}, Activation: {activation_type}, Learning Rate: {learning_rate}, Test Accuracy: {test_acc}')
    ax1.legend()
    ax1.set_xlim((0,10))
    ax1.set_ylim((0,100))

    # bottom subplot for loss trend
    ax2 = plt.subplot(gs[1])
    ax2.plot(epochs, loss_trend, label='Loss Trend', color='green', linewidth = line_width)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Trend')
    ax2.legend()
    ax2.set_xlim((0,10))
    ax2.set_ylim((1,3))

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
    try:
        no_epochs = 10
        min_loss_chng = 0.02
        learning_rates = [0.1,0.01,0.001]
        no_layers = [1,2,3]
        activation_types = [F.relu,F.leaky_relu,F.elu]

        # get the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

                # load the data
                x_train, y_train, x_valid, y_valid,  x_test, y_test = building.getProcessedData()
                train_dataset,val_dataset,test_dataset = building.makeDatasets(x_train,y_train,x_valid, y_valid, x_test,y_test)
                # load onto device
                train_dataset = [(x.to(device), y.to(device)) for x, y in train_dataset]
                val_dataset = [(x.to(device), y.to(device)) for x, y in val_dataset]
                test_dataset = [(x.to(device), y.to(device)) for x, y in test_dataset]
                train_loader, val_loader, test_loader = building.make_DataLoaders(train_dataset,val_dataset,test_dataset,64)

                print("Loaded Data.")

                # initialise pandas dataframe
                df = pd.DataFrame(columns=['activation_function', 'no_layers','learning_rate', 'test_accuracy'])

                # keep track of best models
                best_model_accs = []
                best_model_params = []

                print("Starting training process.")
                for n_l in no_layers:
                    for acc_type in activation_types:
                        for l_r in learning_rates:
                            # create and train model
                            model = building.NumberClassifier(num_layers=n_l,activation=acc_type)
                            model.to(device)
                            validation_trend, training_trend, loss_trend, epochs = building.train_classifier(model,train_loader,val_loader,no_epochs,min_loss_chng,l_r)

                            # print succesfule training
                            print("Trained with " + acc_type.__name__ + " with " + str(n_l) + " layers at learning rate of " + str(l_r))

                            # validate model
                            # acc = building.validate_classifier(model,test_loader)
                            acc = building.validate_jpgs(model,device,'data/hand_crafted')

                            # save trends to a csv
                            save_trends(model,validation_trend,training_trend,loss_trend,epochs,acc,n_l,acc_type,l_r)

                            # add data to the pandas df
                            record = {
                                    'activation_function': acc_type.__name__,
                                    'no_layers': n_l,
                                    'learning_rate': l_r,
                                    'test_accuracy': acc
                                }
                            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
                            df.to_csv("results/test_accuracies.csv", index=False)

                            # check if its one of the best models

        traverse_and_plot("results/trends")
        subprocess.run(['python3','scripts/plotting.py'])
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")
        torch.cuda.empty_cache()
        print("GPU memory released.")
        sys.exit(1)