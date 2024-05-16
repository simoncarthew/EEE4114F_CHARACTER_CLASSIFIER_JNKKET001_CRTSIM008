import matplotlib.pyplot as plt
import shutil
import pandas as pd
import csv
import os
import subprocess

def read_plots_and_combine(df):
    df = df.sort_values(by=['activation_function', 'no_layers', 'learning_rate'], ascending=[True, True,False])
    
    dest_plot = ""
    col = 0
    for index, row in df.iterrows():
        # get .png name
        plot_name = "results/plots/test_"
        plot_name += str(row['no_layers']) + "_"
        plot_name += str(row['activation_function']).replace("_", "") + "_"
        plot_name += str(float(row['learning_rate']))[2:] + "_"
        plot_name += str(float(round(row['test_accuracy'],2))).replace(".","_") + ".png"
        
        if col == 0:
            dest_plot = "results/combined_plots/" + row['activation_function'] + "_" + str(row['no_layers']) + ".png"
            shutil.copy(plot_name, dest_plot)
        else:
            subprocess.run(['convert', '+append'] + [dest_plot,plot_name] + [dest_plot])
        
        col += 1

        if col == 3:
            col = 0


def plot_performance(df):
    for act_func in df['activation_function'].unique().tolist():
        plt.figure()

        for lr in df['learning_rate'].unique().tolist():
            df_filt = df[(df['activation_function'] == act_func) & (df['learning_rate'] == lr)]
            plt.plot(df_filt['no_layers'], df_filt['test_accuracy'],label="LR = " + str(lr))
        
        plt.title(f'Test Accuracy vs Number of Layers for {act_func.capitalize()}')
        plt.xlabel('Number of Layers')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/combined_plots/" + act_func + ".png")
        plt.close()

if __name__ == "__main__":
    df = pd.read_csv("results/test_accuracies.csv")
    read_plots_and_combine(df)
    plot_performance(df)