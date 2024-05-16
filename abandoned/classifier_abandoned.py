#### IMPORTS ####
import torch
import sys
import string
import scripts.building_abandoned as building_abandoned


#### MAIN ####
if __name__ == "__main__":
    try:
        # set constant training hyper parameters
        no_epochs = 30
        min_loss_chng = 0.02
        learn_rate = 0.01
        batch_size = 32
        num_models = 1

        # target model list for multiple model classification
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
        comb_classifier = building_abandoned.CombLetterClassifier(num_models,device,create_models=False,model_targets=positions_list)

        # load last model if exists and user requests it
        loaded = False
        load_new = input("Would you like to load the previous model (y/n):")
        if load_new == "y":
            comb_classifier.load(num_models,device)
            print("Model loaded successfully.")
            loaded = True
            
        # train the model if not loaded
        if loaded == False:

            comb_classifier = building_abandoned.CombLetterClassifier(num_models,device,model_targets=positions_list)

            print("Loading Data.")
            comb_classifier.createDataLoaders("data/kaggle/kaggle_letters.csv",batch_size)
            print("Data Loaded.")

            print("Training Models.")
            comb_classifier.train(no_epochs,min_loss_chng,learn_rate,display=True)

            # ask if the user would like to save the current model
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
                image = building_abandoned.JPGtoTensor(file_path,device)
                pred_class = comb_classifier.classify(image)
                print("Im pretty sure its",alphabet[pred_class])
            file_path = input("Please enter a filepath (\"q\" to quit): ")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")
        torch.cuda.empty_cache()
        print("GPU memory released.")
        sys.exit(1)